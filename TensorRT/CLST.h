#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"

using namespace nvinfer1;
using namespace cv;

struct DetectResult 
{
	int classId;
	float conf;
	cv::Rect box;
};

class CLST
{
public:
	void start_Engine(std::string enginefile, float conf_thresholod, float score_thresholod);
	void segmentation(cv::Mat &frame, std::vector<DetectResult> &results);
	~CLST();
private:
	float sigmoid_function(float a);
	float conf_thresholod = 0.25;
	float score_thresholod = 0.25;
	int input_h = 640;
	int input_w = 640;
	int output_h;
	int output_w;
	IRuntime* runtime{ nullptr };
	ICudaEngine* engine{ nullptr };
	IExecutionContext* context{ nullptr };
	void* buffers[3] = { NULL, NULL , NULL };
	std::vector<float> prob;
	std::vector<float> mprob; // mask
	cudaStream_t stream;
};