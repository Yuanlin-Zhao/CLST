#include "CLST.h"

#pragma comment(lib, "./CLST.lib")

int main(int argc, char** argv) {
	std::vector<std::string> labels;
	labels.push_back("crack");
	std::string enginefile = "./model.engine";
	cv::VideoCapture cap(0);
	cv::Mat frame;
	auto detector = std::make_shared<CLST>();
	detector->start_Engine(enginefile, 0.25, 0.25);
	std::vector<DetectResult> results;
	while (true) {
		bool ret = cap.read(frame);
		if (frame.empty()) {
			break;
		}
		detector->segmentation(frame, results);
		for (DetectResult dr : results) {
			cv::Rect box = dr.box;
			cv::putText(frame, labels[dr.classId] + std::to_string(dr.conf), cv::Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 0, 0));
		}
		cv::imshow("CLST", frame);
		cv::waitKey(1);

		results.clear();
	}
	return 0;
}