#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace fsm {

	class ShapeModel; //3rd class warpper

	struct Match
	{
		uint32_t templ_id;
		float score;
		float angle;
		float scale;
		cv::Point2f match_center; //train image center

		cv::Point2i match_pos;
		cv::Point2i match_box_tl;
		//cv::Size train_size;

		cv::Size match_box_size;
		std::string class_id;
		std::vector<cv::Point2f> match_feat_points;
	};

	class CreateShapeModel
	{
	public:
		CreateShapeModel();
		~CreateShapeModel();
		void create(cv::Mat img);
	public:
		std::string class_id;
		std::string save_path;
		uint32_t feature_num = 64;
		float weak_thresh = 30.0f;
		float strong_thresh = 60.0f;
		float angle_step = 1.0f;
		float scale_step = 1.0f;
		std::vector<int> T_at_level = { 4, 8 };
		std::vector<float> angle_range = { 0.0f };
		std::vector<float> scale_range = { 1.0f };
	};

	class FindShapeModel
	{
	public:
		FindShapeModel();
		~FindShapeModel();
		void loadModel(const std::string& path, const std::vector<std::string>& ids);
		std::vector<Match> find(cv::Mat insp);

	public:
		std::vector<std::string> class_ids;
		std::string templ_path;
		std::vector<int> T_at_level = { 4, 8 }; //金字塔步长
		uint32_t feature_num = { 64 };
		float weak_thresh = { 30.0f };
		float strong_thresh = { 60.0f };
		float score_thresh = { 60.0f };			//最低得分
		uint32_t desired = UINT32_MAX;			//设置想要找到shape的最大数量,不设置输出找到的最大数

	private:
		std::unique_ptr<ShapeModel> _shape;     //使用只需头文件
	};
}
