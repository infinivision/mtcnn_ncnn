#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "mtcnn.h"

void test_video(int argc, char* argv[]) {
	std::string model_path = argv[1];
	MTCNN mm(model_path);
	cv::VideoCapture mVideoCapture(0);
	if (!mVideoCapture.isOpened()) {
		return;
	}
	cv::Mat frame;
	mVideoCapture >> frame;
	while (!frame.empty()) {
		mVideoCapture >> frame;
		if (frame.empty()) {
			break;
		}

		clock_t start_time = clock();

		ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
		std::vector<Bbox> finalBbox;
		mm.detect(ncnn_img, finalBbox);
		const int num_box = finalBbox.size();
		std::vector<cv::Rect> bbox;
		bbox.resize(num_box);
		for(int i = 0; i < num_box; i++){
			bbox[i] = cv::Rect(finalBbox[i].x1, finalBbox[i].y1, finalBbox[i].x2 - finalBbox[i].x1 + 1, finalBbox[i].y2 - finalBbox[i].y1 + 1);
		 }
		for (vector<cv::Rect>::iterator it = bbox.begin(); it != bbox.end(); it++) {
			rectangle(frame, (*it), cv::Scalar(0, 0, 255), 2, 8, 0);
		}

		imshow("face_detection", frame);
		clock_t finish_time = clock();
		double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;
		std::cout << "time" << total_time * 1000 << "ms" << std::endl;

		int q = cv::waitKey(10);
		if (q == 27) {
			break;
		}
	}
	return ;
}

int main(int argc, char* argv[]) {
    test_video(argc, argv);
}