#include "find_shape_model.h"
#include <string>
#include <vector>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

int main()
{
	cv::Mat train_img = cv::imread("D:/test/copy/mark4.jpg");
	fsm::CreateShapeModel createModel;
	createModel.class_id = "ttt";
	createModel.feature_num = 64;
	createModel.T_at_level = { 4, 8 };
	createModel.angle_range = { -5, 5 };
	createModel.save_path = "D:/test/copy";
	//createModel.create(train_img);

	fsm::FindShapeModel findModel;
	std::string path = "D:/test/copy";
	findModel.loadModel(path, { "mark" });
	cv::Mat insp_img = cv::imread("d:/test/copy/insp_roi_res2.jpg");
	auto matches = findModel.find(insp_img);
	std::cout << "matches size = " << matches.size() << std::endl;

	return 0;
}