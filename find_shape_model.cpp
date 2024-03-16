#include "find_shape_model.h"
#include <filesystem>
#include <stdexcept>
#include <unordered_map>
#include <opencv2/dnn/dnn.hpp>
#include "line2Dup.h"
#include "cuda_icp/icp.h"

namespace fsm {
	// single shape train info
	struct ShapeInfos
	{
		std::vector<shape_based_matching::shapeInfo_producer::Info> infos;
		cv::Size size;
	};

	struct ShapeModel
	{
		std::unique_ptr<line2Dup::Detector> detector;
		std::unordered_map<std::string, ShapeInfos> infos_map;
	};
}

fsm::CreateShapeModel::CreateShapeModel()
{

}

fsm::CreateShapeModel::~CreateShapeModel()
{

}

void fsm::CreateShapeModel::create(cv::Mat img)
{
	if (img.empty())
		throw std::invalid_argument("img is empty");
	if (!std::filesystem::exists(save_path))
		throw std::invalid_argument(save_path);
	if (angle_range.size() > 2)
		throw std::invalid_argument("angle range is invalid");
	if (scale_range.size() > 2)
		throw std::invalid_argument("scale range is invalid");

	line2Dup::Detector detector(feature_num, T_at_level, weak_thresh, strong_thresh);
	cv::Mat mask = cv::Mat(img.size(), CV_8UC1, { 255 });

	shape_based_matching::shapeInfo_producer shapes(img, mask);
	shapes.scale_range = scale_range;
	shapes.scale_step = scale_step;
	shapes.angle_range = angle_range;
	shapes.angle_step = angle_step;
	shapes.produce_infos();

	std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
	for (auto& info : shapes.infos) {
		cv::imshow("train", shapes.src_of(info)); 
		cv::waitKey(1);

		std::cout << "\ninfo.angle: " << info.angle << std::endl;
		int templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info));
		std::cout << "templ_id: " << templ_id << std::endl;

		if (templ_id != -1) 
			infos_have_templ.push_back(info);
	}

	detector.writeClasses(save_path + "/%s_templ.yaml");
	std::string info_path = save_path + "/" + class_id + "_info.yaml";
	shapes.save_infos(infos_have_templ, info_path);

	cv::FileStorage info_write(info_path, cv::FileStorage::APPEND);
	std::vector<int> img_size = { img.cols, img.rows };
	info_write << "imageSize" << img_size;

	std::cout << "train end" << std::endl << std::endl;
}


/////////////////////
fsm::FindShapeModel::FindShapeModel()
{
}

fsm::FindShapeModel::~FindShapeModel()
{
}

void fsm::FindShapeModel::loadModel(const std::string& path, const std::vector<std::string>& ids)
{
	if (!std::filesystem::exists(path))
		throw std::invalid_argument(path);
	_shape = std::make_unique<fsm::ShapeModel>();

	_shape->detector = std::make_unique<line2Dup::Detector>(feature_num, T_at_level, weak_thresh, strong_thresh);
	_shape->detector->readClasses(ids, path + "/%s_templ.yaml");

	auto loadShapeInfos = [this, &path](const std::string& id) {
		ShapeInfos shapeInfos;
		//auto filename = cv::format("%s/%s_info.yaml", path, id); error
		std::string filename = path + "/" + id + "_info.yaml";
		shapeInfos.infos = shape_based_matching::shapeInfo_producer::load_infos(filename);

		std::vector<int> size;	/*训练图片的Size*/
		cv::FileStorage fs_read(filename, cv::FileStorage::READ);
		fs_read["imageSize"] >> size;
		shapeInfos.size = cv::Size(size.at(0), size.at(1));

		_shape->infos_map.insert_or_assign(id, shapeInfos);
		};

	for (auto& id : ids)
		loadShapeInfos(id);
}

std::vector<fsm::Match> fsm::FindShapeModel::find(cv::Mat insp)
{
	if (_shape->detector == nullptr)
		throw std::runtime_error("detector is nullptr!");

	/*produce dxy, for icp purpose maybe*/
	_shape->detector->set_produce_dxy = true;
	std::vector<line2Dup::Match> matches = _shape->detector->match(insp, score_thresh, class_ids);

	//use icp to match
	// use kdtree to query, expected to be faster
	Scene_kdtree scene;
	KDTree_cpu kdtree;
	scene.init_Scene_kdtree_cpu(_shape->detector->dx_, _shape->detector->dy_, kdtree);

	auto icpTransform = [](const cuda_icp::RegistrationResult& result, const cv::Point2f& point) {
		cv::Point2f new_point;
		new_point.x = result.transformation_[0][0] * point.x + result.transformation_[0][1] * point.y + result.transformation_[0][2];
		new_point.y = result.transformation_[1][0] * point.x + result.transformation_[1][1] * point.y + result.transformation_[1][2];
		return new_point;
		};

	auto exportMatch = [this, &scene, &icpTransform](const line2Dup::Match& match) {
		fsm::Match fsm_match;	//store result
		auto templ = _shape->detector->getTemplates(match.class_id, match.template_id); //寻找匹配的特征模板

		std::vector<::Vec2f> model_pcd;
		model_pcd.reserve(templ[0].features.size());
		for (auto& feat : templ[0].features)
			model_pcd.emplace_back(float(feat.x + match.x), float(feat.y + match.y));

		// subpixel, also refine scale
		cuda_icp::RegistrationResult result = cuda_icp::sim3::ICP2D_Point2Plane_cpu(model_pcd, scene);

		//transform icp features
		fsm_match.match_feat_points.reserve(model_pcd.size());
		for (auto& feat : templ[0].features) {
			auto icp_point = icpTransform(result, { float(feat.x + match.x) ,float(feat.y + match.y) });
			fsm_match.match_feat_points.emplace_back(icp_point);
		}
		//transform icp train img center
		auto train_img_half_width = _shape->infos_map.at(match.class_id).size.width / 2.0;
		auto train_img_half_height = _shape->infos_map.at(match.class_id).size.height / 2.0;
		float cx = match.x - templ[0].tl_x + train_img_half_width;
		float cy = match.y - templ[0].tl_y + train_img_half_height;
		fsm_match.match_center = icpTransform(result, {cx, cy});
		//transform icp angle
		/*icp angle*/
		double init_angle = _shape->infos_map.at(match.class_id).infos[match.template_id].angle;
		init_angle = init_angle >= 180 ? (init_angle - 360) : init_angle;
		double ori_diff_angle = std::abs(init_angle);
		double icp_diff_angle = std::abs(-std::asin(result.transformation_[1][0]) / CV_PI * 180 + init_angle);
		double improved_angle = ori_diff_angle - icp_diff_angle;
		fsm_match.angle = improved_angle;

		fsm_match.class_id = match.class_id;
		fsm_match.score = match.similarity;
		fsm_match.match_pos = { match.x, match.y };
		fsm_match.templ_id = match.template_id;
		fsm_match.scale = _shape->infos_map.at(match.class_id).infos[match.template_id].scale;
		fsm_match.match_box_tl = { templ[0].tl_x, templ[0].tl_y };
		fsm_match.match_box_size = { templ[0].width, templ[0].height };

		return fsm_match;
		};

	auto NMSMatches = [this](const std::vector<line2Dup::Match>& matches) {
		std::vector<cv::Rect> boxes;
		std::vector<float> scores;
		boxes.reserve(matches.size());
		scores.reserve(matches.size());
		for (const auto& match : matches)
		{
			auto templ = _shape->detector->getTemplates(match.class_id, match.template_id);
			boxes.emplace_back(match.x, match.y, templ[0].width, templ[0].height);
			scores.push_back(match.similarity);
		}
		std::vector<int> idxs;
		cv::dnn::NMSBoxes(boxes, scores, 0, 0.5f, idxs);
		return idxs;
		};

	auto idxs = NMSMatches(matches);
	/*max num of shape expected to find*/
	int top = (desired > idxs.size()) ? idxs.size() : desired;

	std::vector<fsm::Match> icp_matches;
	icp_matches.reserve(top);
	for (auto idx : idxs)
	{
		auto icp_match = exportMatch(matches[idx]);
		icp_matches.emplace_back(icp_match);
	}
	return icp_matches;
}