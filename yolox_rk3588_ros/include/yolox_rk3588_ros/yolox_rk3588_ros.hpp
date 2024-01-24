#pragma once

#include <yolox_rk3588_cpp/yolox.hpp>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection2_d.hpp>

namespace yolox_rk3588_ros
{
class YOLOX_ROS : public rclcpp::Node
{
public:
    explicit YOLOX_ROS(rclcpp::NodeOptions options);
    ~YOLOX_ROS();

    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr &msg);

private:
    yolox_rk3588_cpp::YOLOX *yolox;
    bool imshow_is_show = false;
    
    image_transport::Subscriber image_sub;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_pub;
};

} // namespace yolox_rk3588_ros