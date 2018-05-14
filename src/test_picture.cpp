#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "mtcnn.h"


int test_picture(int argc, char** argv) {
	std::string model_path = argv[1];
	MTCNN mm(model_path);

	std::cout << "after load model..." << std::endl;
	clock_t start_time = clock();

	cv::Mat image;
    std::string image_path = argv[2];
	image = cv::imread(image_path);
	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
	std::vector<Bbox> finalBbox;
	mm.detect(ncnn_img, finalBbox);

	const int num_box = finalBbox.size();
	cout << "num_box: " << num_box << endl;
	std::vector<cv::Rect> bbox;
	bbox.resize(num_box);
	for (int i = 0; i < num_box; i++) {
		bbox[i] = cv::Rect(finalBbox[i].x1, finalBbox[i].y1, finalBbox[i].x2 - finalBbox[i].x1 + 1, finalBbox[i].y2 - finalBbox[i].y1 + 1);

	}
	for (vector<cv::Rect>::iterator it = bbox.begin(); it != bbox.end(); it++) {
		rectangle(image, (*it), cv::Scalar(0, 0, 255), 2, 8, 0);
	}

	std::cout << "bbox size: " << bbox.size() << std::endl;

	exit(0);
	imshow("face_detection", image);
	clock_t finish_time = clock();
	double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;
	std::cout << "time" << total_time * 1000 << "ms" << std::endl;

	cv::waitKey(0);

}

int main(int argc, char* argv[]) {
    test_picture(argc, argv);
}
