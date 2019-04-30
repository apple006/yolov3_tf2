# TensorFlow2.0 从零实现YoloV3检测网络

前言： 很久之前开源过一个基于fastercnn的kitti目标检测模型，直到现在都有人star它并通过微信联系到我，向我请教一些疑问。现如今各种检测算法层出不穷，甚至像FoveaBox, CenterNet等新生代方法已经快要摒弃传统的（是的，没有错，深度学习方法也会沦为传统方法）检测方法。在这种情况下，我觉得有必要对所有的检测网络进行一个总结，并再次以传教士的身份传授更多的入门者当下最先进的检测网络。我要做的第一件事情，就是跟上Tensorflow的步伐，完成这个基于tensorflow2.0 全新API的实现YoloV3的教程。

尽管到现在，YoloV3依然是STOA的检测算法，最大的优点在于它的速度更快，精度比其他一阶段检测方法都好。在开始之前，有几条原则必须要铭记：

- No fancy. 写代码的时候，别搞各种fancy的东西，网络设计你可以fancy，简单的代码别；
- 保持原生。这一点相信很多以前的tensorflow用户深受诟病，那么再接下来的开发中，请铭记这一条原则，请记住，别人的wrapper不靠谱，也没有必要；
- Handle everything. 这一点很重要，很多刚入门的人在写一个训练脚本，根本不会考虑暂停、断点续训、再exception发生的时候及时保存权重，如果你刚好看了这篇教程入门AI，请牢记这条原则，你需要考虑所有可能发生的情况。
- 产品思维。写代码的时候，请记住你的代码是面向产品的，所有速度太慢的算法别用，自己写的太垃圾的算法请寻找别人STOA代替，请遵照pep8标准写代码，请时刻想到怎么能让用户使用方便。


OK，那么开始吧。

## 1. 从Dataloader说起

很多人学AI，上来就是`import tensorflow as tf`. 其实没有必要，我建议大家先把数据预处理的本领先学会了。比如数据你怎么read？你怎么resize，怎么padding，怎么convert bbox, 怎么normalize等等，至今为止，我还从未见过一篇讲解这些东西的文章，甚至没有人告诉大家伙这些东西的重要性。别说你觉得这个太简单，那请思考一下voc的图片mask的label是保存的几位的数据？像素值是从多少到多少？没有人从基础去处理就肯定不知道。当然你可能知道它是8位的像素值，那么再次请问为什么8位它能够显示成彩色？依旧无人深入去思考这个问题。关于这个问题我之前有一篇博客探讨过，大家可以前往翻一翻(http://jinfagang.github.io).

我们是来实现YoloV3的，说到dataload，我想告诉大家第一个干货：**使用tensorflow官方的datasets库**. 如果你不看本教程，这个大部分人肯定不知道，来看看人家tensorflow给你准备了多少数据集的dataset类：

![](https://s2.ax1x.com/2019/04/30/EGJjud.md.png)

在load coco之前，先从mnist来体验一下tfds (先安装一下tensorflow_datasets)：

```python
import tensorflow as tf
import tensorflow_datasets as tfds

# tfds works in both Eager and Graph modes
tf.enable_eager_execution()

# See available datasets
print(tfds.list_builders())

# Construct a tf.data.Dataset
dataset = tfds.load(name="mnist", split=tfds.Split.TRAIN)

# Build your input pipeline
dataset = dataset.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
for features in dataset.take(1):
  image, label = features["image"], features["label"]
```