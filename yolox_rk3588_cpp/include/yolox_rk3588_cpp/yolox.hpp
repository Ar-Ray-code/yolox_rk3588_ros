// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
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


int init_yolox_model(const char* model_path, rknn_app_context_t* app_ctx);
int release_yolox_model(rknn_app_context_t* app_ctx);
int inference_yolox_model(rknn_app_context_t* app_ctx, image_buffer_t* img, object_detect_result_list* od_results);

cv::Mat3b image_buffer_to_mat3b(image_buffer_t* image_buffer);
image_buffer_t mat3b_to_image_buffer(cv::Mat3b* mat);
void show_and_write(cv::Mat3b* mat, object_detect_result_list* od_results, const char* image_path);