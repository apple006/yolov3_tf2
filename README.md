# YoloV3 in TensorFlow 2.0

Enhanced version of original implementation from [here](https://github.com/zzh8829/yolov3-tf2). This is a detailed and simple enough implementation of YoloV3, and it is written in *TensorFlow 2.0* ! We did those improvements compare to previous version:

- Add MobileNetV2 backbone, so it supports YoloV3 MobileNetV2;
- DCN network implemented;
- Add un-same-width image input (wider images);
- Add training on KITTI and nuScenes, also a TrafficLight detection model;


## Install

there are several requirements for this repo:

```
absl-py
alfred-py
loguru
pytorch
```

If you got some package missing error, just install it.

## More

We plan to do more things in the near future:

- Adding nms gpu in C++ and cuda;
- Accelerate the network using pruning and compressing model;
- Export to onnx and convert to ncnn.


## Copyright

this work original implemented by `zzh8829`, enhanced by `jinfagang`. authors got there right respectively.
