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

    // init rknn
    yolox_rk3588_cpp::YOLOX *yolox = new yolox_rk3588_cpp::YOLOX(model_path);

    // inference
    cv::Mat3b img = cv::imread(image_path);
    object_detect_result_list od_results = yolox->inference(&img);
    yolox->show_and_write(&img, &od_results, image_path);

    // deinit rknn
    delete yolox;

    return 0;
}
