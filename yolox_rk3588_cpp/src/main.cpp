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

#include "yolox_rk3588_cpp/yolox.hpp"


int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("%s <model_path> <image_path>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *image_path = argv[2];

    cv::Mat3b img = cv::imread(image_path);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();

    if (init_yolox_model(model_path, &rknn_app_ctx) != 0)
    {
        printf("init_yolox_model fail!\n");
        return -1;
    }

    image_buffer_t src_image;
    object_detect_result_list od_results;

    memset(&src_image, 0, sizeof(image_buffer_t));
    src_image = mat3b_to_image_buffer(&img);

    if (inference_yolox_model(&rknn_app_ctx, &src_image, &od_results) != 0)
    {
        printf("init_yolox_model fail!\n");
        return -1;
    }
    show_and_write(&img, &od_results, image_path);
    deinit_post_process();

    if (release_yolox_model(&rknn_app_ctx) != 0)
        printf("release_yolox_model fail!\n");

    if (src_image.virt_addr != NULL)
        free(src_image.virt_addr);

    return 0;
}
