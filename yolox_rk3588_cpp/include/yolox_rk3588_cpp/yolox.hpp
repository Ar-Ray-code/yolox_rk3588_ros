#pragma once

#include <cmath>
#include <cstring>
#include <cstdio>
#include <math.h>
#include <sys/time.h>
#include <set>
#include <string>
#include <vector>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "yolox_rk3588_cpp/rknpu2/rknn_api.h"
#include "yolox_rk3588_cpp/common.hpp"
#include "yolox_rk3588_cpp/image_utils.hpp"
#include "yolox_rk3588_cpp/file_utils.hpp"


typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
} rknn_app_context_t;
#include "yolox_rk3588_cpp/postprocess.hpp"

namespace yolox_rk3588_cpp
{

class YOLOX
{

public:
    explicit YOLOX(std::string model_path);
    ~YOLOX();

    int init_yolox_model(const char* model_path, rknn_app_context_t* app_ctx);
    int release_yolox_model(rknn_app_context_t* app_ctx);
    int inference_yolox_model(rknn_app_context_t* app_ctx, image_buffer_t* img, object_detect_result_list* od_results);

    cv::Mat3b image_buffer_to_mat3b(image_buffer_t* image_buffer);
    image_buffer_t mat3b_to_image_buffer(cv::Mat3b* mat);
    void show_and_write(cv::Mat3b* mat, object_detect_result_list* od_results, const char* image_path);

    object_detect_result_list inference(cv::Mat3b* mat);

private:
    rknn_app_context_t rknn_app_ctx;
};

} // namespace yolox_rk3588_cpp
