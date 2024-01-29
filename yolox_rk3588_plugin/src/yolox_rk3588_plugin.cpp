// Copyright 2024 StrayedCats.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "yolox_rk3588_plugin/yolox_rk3588_plugin.hpp"

namespace detector2d_plugins
{

void YoloxRk3588::init(const detector2d_parameters::ParamListener & param_listener)
{
  params_ = param_listener.get_params();
  yolo = new yolox_rk3588_cpp::YOLOX(this->params_.model_path);
  std::cout << "model loaded : " << this->params_.model_path << std::endl;
}

Detection2DArray YoloxRk3588::detect(const cv::Mat & image)
{
  // mat to mat3b
  cv::Mat3b image3b = image;
  std::cout << "YoloxRk3588::detect1" << std::endl;

  object_detect_result_list od_results = yolo->inference(&image3b);

  vision_msgs::msg::Detection2DArray detection_array;
  for (int i = 0; i < od_results.count; i++)
  {
    object_detect_result* det_result = &(od_results.results[i]);
    std::string class_id = yolox_rk3588_cpp::COCO_CLASSES[det_result->cls_id];

    vision_msgs::msg::Detection2D detection;
    detection.bbox.center.position.x = det_result->box.left;
    detection.bbox.center.position.y = det_result->box.top;
    detection.bbox.size_x = det_result->box.right - det_result->box.left;
    detection.bbox.size_y = det_result->box.bottom - det_result->box.top;
    detection.id = class_id;
    detection.results.resize(1);
    detection.results[0].hypothesis.class_id = class_id;
    detection.results[0].hypothesis.score = det_result->prop;
    detection_array.detections.push_back(detection);
  }

  return detection_array;
}

}// namespace detector2d_plugins

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(detector2d_plugins::YoloxRk3588, detector2d_base::Detector)