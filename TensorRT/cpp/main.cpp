#include "CLST.h"

#pragma comment(lib, "./CLST.lib")
std::string labels_txt_file = "";
std::vector<std::string> readClassNames();
std::vector<std::string> readClassNames()
{
	std::vector<std::string> classNames;

	std::ifstream fp(labels_txt_file);
	if (!fp.is_open())
	{
		printf("could not open file...\n");
		exit(-1);
	}
	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name);
	}
	fp.close();
	return classNames;
}

int main(int argc, char** argv) {
	std::vector<std::string> labels = readClassNames();
	std::string enginefile = "";
	cv::VideoCapture cap("");
	cv::Mat frame;
	auto detector = std::make_shared<CLST>();
	detector->initConfig(enginefile, 0.25, 0.25);
	std::vector<DetectResult> results;
	while (true) {
		bool ret = cap.read(frame);
		if (frame.empty()) {
			break;
		}
		detector->detect(frame, results);
		for (DetectResult dr : results) {
			cv::Rect box = dr.box;
			cv::putText(frame, labels[dr.classId] + std::to_string(dr.conf), cv::Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 0, 0));
		}
		cv::imshow("CLST + TensorRT 实例分割演示", frame);
		char c = cv::waitKey(1);
		if (c == 27) { // ESC 退出
			break;
		}
		// reset for next frame
		results.clear();
	}
	return 0;
}