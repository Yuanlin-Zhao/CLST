import torch
import torchvision
import torch.nn as nn
import numpy as np


##############################################################Backbone#######################################################
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):

    default_act = nn.GELU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):

        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):

        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):

        return self.act(self.conv(x))

class RepConv(nn.Module):
    default_act = nn.GELU()
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process"""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process"""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.c1
        groups = self.g
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')

class RepBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(RepBlock, self).__init__()

        self.conv1 = Conv(inplanes, planes * 2, k=1, s=1)
        self.conv2 = RepConv(planes * 2, planes * 2, k=3)

        self.outconv = nn.Conv2d(planes * 2, planes, kernel_size=3, stride=2, padding=6, dilation=6)

        self.inconv = nn.Sequential(nn.Conv2d(inplanes, planes * 2, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(planes * 2),
                                    nn.GELU())

    def forward(self, x):
        identity = x
        identity = self.inconv(identity)
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        out = self.outconv(out)
        return out


class SplitTransformerLayer(nn.Module):
    def __init__(self, c, num_heads):
        super().__init__()

        self.q = nn.Linear(c // 4, c // 4, bias=False)
        self.k = nn.Linear(c // 4, c // 4, bias=False)
        self.v = nn.Linear(c // 4, c // 4, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c // 4, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)
        self.bnSplit = nn.BatchNorm1d(c // 4)
        self.actSplit = nn.GELU()
        self.bn = nn.BatchNorm1d(c)
        self.act = nn.GELU()

    def forward(self, x):
        SplitC = x.size()[2] // 4

        x1, x2, x3, x4 = torch.split(x, [SplitC, SplitC, SplitC, SplitC], dim=2)

        "Four-way multi-head processing"
        x1 = self.actSplit(self.bnSplit(self.ma(self.q(x1), self.k(x1), self.v(x1))[0].permute(0, 2, 1))).permute(0, 2, 1) + x1
        x2 = self.actSplit(self.bnSplit(self.ma(self.q(x2), self.k(x2), self.v(x2))[0].permute(0, 2, 1))).permute(0, 2, 1) + x2
        x3 = self.actSplit(self.bnSplit(self.ma(self.q(x3), self.k(x3), self.v(x3))[0].permute(0, 2, 1))).permute(0, 2, 1) + x3
        x4 = self.actSplit(self.bnSplit(self.ma(self.q(x4), self.k(x4), self.v(x4))[0].permute(0, 2, 1))).permute(0, 2, 1) + x4

        x = torch.cat([x1, x2, x3, x4], dim=2)

        return self.fc2(self.fc1(x)) + x

class CrackLayer(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.input_dim = c1
        self.output_dim = c2
        self.norm = nn.LayerNorm(c1)
        self.ST = SplitTransformerLayer(c1 // 4, 4)
        self.proj = nn.Linear(c2, c2)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):

        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_ST1 = self.ST(x1) + self.skip_scale * x1
        x_ST2 = self.ST(x2) + self.skip_scale * x2
        x_ST3 = self.ST(x3) + self.skip_scale * x3
        x_ST4 = self.ST(x4) + self.skip_scale * x4
        x_ST = torch.cat([x_ST1, x_ST2, x_ST3, x_ST4], dim=2)

        x_ST = self.norm(x_ST)
        x_ST = self.proj(x_ST)

        out = x_ST.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)

        return out

class DSPPF(nn.Module):

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1, d=6)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2 )

    def forward(self, x):

        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)

        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class RepCrackFormer(nn.Module):
    def __init__(self, c1, c2, mode='s'):
        super().__init__()

        self.mode = mode
        self.c1 = c1  #c1 = 256
        self.c2 = c2

        if self.mode == 's':
            "The reparameter downsampling"
            self.RepDown = RepConv(c1, c1 * 2, k=3, s=2)
            "The CrackLayer"
            self.CrackLayer = nn.Sequential(CrackLayer(c1 * 2, c1 * 2))
            "The Dilated Feature Pyramid"
            self.FeaturePyramid = DSPPF(c1 * 2, c2)

        if self.mode == 'x':
            "The Reparameter Downsampling"
            self.RepDown = RepBlock(c1, c1 * 2)
            "The CrackLayer"
            self.CrackLayer = nn.Sequential(CrackLayer(c1 * 2, c1 * 2))
            "The Dilated Feature Pyramid"
            self.FeaturePyramid = DSPPF(c1 * 2, c2)

    def forward(self, x):

        x = self.RepDown(x)
        x = self.CrackLayer(x)
        x = self.FeaturePyramid(x)

        return x



##############################################################Neck#######################################################

class CrackConv(nn.Module):

    default_act = nn.GELU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=6, act=True):
        super().__init__()

        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=1, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):

        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):

        return self.act(self.conv(x))

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()

        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):

        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()

        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):

        return x * self.sigmoid(x)

class CoordinateModule(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordinateModule, self).__init__()

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
class Location(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(CoordinateModule(self.c) for _ in range(n))
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))

        return y
from ultralytics.nn.modules import C2f
class CrackBottleNeck(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, n=1, p=None, g=1, d=1, short=False):
        super().__init__()
        self.downconv = Conv(c1=c1, c2=c2, k=k, s=s, g=g, d=d, act=True)
        self.bottleneck = C2f(c2, c2, n, short)

    def forward(self, x):
        return self.bottleneck(self.downconv(x))