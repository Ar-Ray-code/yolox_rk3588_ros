# yolox_rk3588
YOLOX + RK3588

## Depends

[Ubuntu22.04 for Rockchip](https://github.com/Joshua-Riek/ubuntu-rockchip)

## INT8 Speed

| Model | FPS |
| --- | --- |
| YOLOX-Nano | 58 |
| YOLOX-Tiny | 52 |
| YOLOX-S | 23 |
| YOLOX-M | |

```text
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib"
```


```bash
ros2 run yolox_rk3588_ros yolox_rk3588_ros_node  --ros-args -p model_path:=/home/rock5a/yolox_s_default.rknn
```

## Ref

- [airockchip/rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo)
