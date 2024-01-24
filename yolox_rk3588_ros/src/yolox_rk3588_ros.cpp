#include <yolox_rk3588_ros/yolox_rk3588_ros.hpp>

namespace yolox_rk3588_ros
{

YOLOX_ROS::YOLOX_ROS(rclcpp::NodeOptions options) :
    Node("yolox_rk3588_ros", options)
{
    std::string model_path;

    this->declare_parameter<std::string>("model_path", "/home/rock5a/yolox_tiny_default.rknn");
    this->declare_parameter<bool>("imshow_is_show", true);

    this->get_parameter("model_path", model_path);
    this->get_parameter("imshow_is_show", imshow_is_show);

    yolox = new yolox_rk3588_cpp::YOLOX(model_path);

    image_sub = image_transport::create_subscription(this, "image_raw", std::bind(&YOLOX_ROS::image_callback, this, std::placeholders::_1), "raw", rmw_qos_profile_sensor_data);
    detection_pub = this->create_publisher<vision_msgs::msg::Detection2DArray>("detection", 10);
}

YOLOX_ROS::~YOLOX_ROS()
{
    delete yolox;
}

void YOLOX_ROS::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat3b mat = cv_ptr->image;

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    object_detect_result_list od_results = yolox->inference(&mat);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    RCLCPP_INFO(this->get_logger(), "YOLOX inference time: %f ms (%f fps)", time_used.count() * 1000, 1.0 / time_used.count());

    vision_msgs::msg::Detection2DArray detection_array;
    detection_array.header = msg->header;
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

        if (imshow_is_show)
        {
            cv::rectangle(mat, cv::Point(det_result->box.left, det_result->box.top), cv::Point(det_result->box.right, det_result->box.bottom), cv::Scalar(0, 255, 0), 2);
            cv::putText(mat, class_id, cv::Point(det_result->box.left, det_result->box.top), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        }
    }
    detection_pub->publish(detection_array);

    if (imshow_is_show)
    {
        cv::imshow("YOLOX_ROS", mat);
        cv::waitKey(1);
    }
}

} // namespace yolox_rk3588_ros

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(yolox_rk3588_ros::YOLOX_ROS)
