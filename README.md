# YoloV3 in TensorFlow 2.0

Enhanced version of original implementation from [here](https://github.com/zzh8829/yolov3-tf2). This is a detailed and simple enough implementation of YoloV3, and it is written in *TensorFlow 2.0* ! We did those improvements compare to previous version:

- Add MobileNetV2 backbone, so it supports YoloV3 MobileNetV2;
- DCN network implemented;
- Add un-same-width image input (wider images);
- Add training on KITTI and nuScenes, also a TrafficLight detection model;

Welcome subscriber our ZhihuZhuanlan for more detailed Chinese tutorials! https://zhuanlan.zhihu.com/tensorflow2 .


## Install

there are several requirements for this repo:

```
absl-py
alfred-py
loguru
pytorch
```

If you got some package missing error, just install it.

## Training

After you prepared tfrecord file, you can kick off training easily:

```
INFO 05-09 16:28:23 train.py:51 - using YoloV3 model.
INFO 05-09 16:28:29 train.py:58 - loading dataset from: /media/jintain/wd/permenant/datasets/coco/*.tfrecord
INFO 05-09 16:28:33 train.py:113 - model resumed from: ./checkpoints/yolov3_coco-1.ckpt, start at epoch: 1
INFO 05-09 16:28:40 train.py:140 - Epoch: 1, iter: 0, total_loss: 1202.7585, pred_loss: [709.7193, 241.0539, 237.69351]
INFO 05-09 16:28:46 train.py:140 - Epoch: 1, iter: 10, total_loss: 886.7888, pred_loss: [368.6222, 253.66756, 249.7427]
INFO 05-09 16:28:52 train.py:140 - Epoch: 1, iter: 20, total_loss: 1315.7472, pred_loss: [639.7981, 431.42947, 229.50122]
INFO 05-09 16:28:58 train.py:140 - Epoch: 1, iter: 30, total_loss: 658.0850, pred_loss: [110.5262, 140.95578, 391.42163]
```
As you may notice, the training process if very shock in the start of training time. We will update some results after training.
After trained serveral epochs, there were no evidence that the model is converge. So I think there must be something goes wrong... Try to fix that...


## More

We plan to do more things in the near future:

- Adding nms gpu in C++ and cuda;
- Accelerate the network using pruning and compressing model;
- Export to onnx and convert to ncnn.


## Copyright

this work original implemented by `zzh8829`, enhanced by `jinfagang`. authors got there right respectively.
