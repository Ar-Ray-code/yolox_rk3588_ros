#pragma once

#include <stdint.h>
#include <vector>
#include "yolox_rk3588_cpp/rknpu2/rknn_api.h"
#include "yolox_rk3588_cpp/rknn_type.hpp"
#include "yolox_rk3588_cpp/common.hpp"
#include "yolox_rk3588_cpp/image_utils.hpp"

#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define OBJ_CLASS_NUM 6
#define NMS_THRESH 0.25
#define BOX_THRESH 0.15

// class rknn_app_context_t;

typedef struct {
    image_rect_t box;
    float prop;
    int cls_id;
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

int init_post_process();
void deinit_post_process();
char *coco_cls_to_name(int cls_id);
int post_process(rknn_app_context_t *app_ctx, rknn_output *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results);

void deinitPostProcess();
