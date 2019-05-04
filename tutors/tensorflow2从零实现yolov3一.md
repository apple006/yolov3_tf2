# TensorFlow2.0 从零实现YoloV3检测网络

在正式开始教程之前，需要强调一下，**这不仅仅是一篇教你从零实现一个yolov3检测器的教程，同时也是一个最新最详尽比较权威中肯的TensorFlow2.0教程（我们会包含从dataloader到基础keras api网络搭建的所有过程）**. 同时欢迎大家来我们的论坛探讨AI问题: ft.manaai.cn, 同时也欢迎大家支持一下我们做的AI市场平台，海量AI算法开箱即用：http://manaai.cn .

**NOTE** 以下教程全部基于tensorflow2.0着手，如果你还没有安装，请先通过以下命令来安装, 我们假设我们都是用python3，这也是你应该使用的版本，并且如果你拥有GPU希望你能安装CUDA10版本：

```
sudo pip3 install tf-nightly-2.0-preview
# 如果无法找到每日构建版本，试一下下面的（每日构建版本不会覆盖所有平台）
sudo pip3 install tensorflow==2.0.0-alpha0
```
当然，本教程写于2019.05.01, 此时tensorflow2.0还没有正式发布，但可能你看到本教程是已经发布了正式版，相应安装即可，下面的一些接口应该不会有太大的变化。

很久之前开源过一个基于fastercnn的kitti目标检测模型，直到现在都有人star它并通过微信联系到我，向我请教一些疑问。现如今各种检测算法层出不穷，甚至像FoveaBox, CenterNet等新生代方法已经快要摒弃传统的（是的，没有错，深度学习方法也会沦为传统方法）检测方法。在这种情况下，我觉得有必要对所有的检测网络进行一个总结，并再次以传教士的身份传授更多的入门者当下最先进的检测网络。我要做的第一件事情，就是跟上TensorFlow的步伐，完成这个基于tensorflow2.0 全新API的实现YoloV3的教程。

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

tf.enable_eager_execution()

dataset = tfds.load(name="mnist", split=tfds.Split.TRAIN)

# Build your input pipeline
dataset = dataset.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
for features in dataset.take(1):
  image, label = features["image"], features["label"]
```

上面简单的演示了一下如何从tensorflow导入数据，当然这还只是第一步，还没有涉及到如何准备自己的数据，但是请大家记住两点：

- 标准的数据导入方式，最好使用`tf.data.Dataset` API;
- 不要使用老版本的一些API，比如上面的eager模式其实没有必要，更不需要从contrib import，因为2.0没有contrib了, 2.0以下还是得加。

以前你可能会看到一些采用 `make_one_shot_iter()` 这样的方法导入数据。同样，这个方法也被弃用了，直接用 `take(1)`.

OK， 让我们来导入一个稍微复杂一点的分类数据吧，用fashion-mnist：

```python
import tensorflow as tf
import tensorflow_datasets as tfds

dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
print(metadata)
train_dataset, test_dataset = dataset['train'], dataset['test']
train_dataset = train_dataset.shuffle(100).batch(12).repeat()

for img, label in train_dataset.take(1):
    img = img.numpy()
    print(img.shape)
    print(img)
```

我这里尽量的精简代码。这基本上就涵盖了我们需要的所有步骤，我们可以归纳一下，在最新版本的tensorflow中, 如何构建一个自己的dataset loader：

- 所有的data最终都是一个 `tf.data.Dataset` ；
- 熟悉 `tf.data.Dataset` 的一些相关操作比如shuffle，repeat等可以简单的对数据进行一些处理；
- 在for循环里面就可以拿到你需要的input和label；
  
在上面这个例子中，你就可以拿到你需要的数据的了（每一个epoch训练所需）：

```
(12, 28, 28, 1)
[[[[  0]
   [  0]
   [  0]
   ...
   [  0]
   [  0]
   [  0]]

  [[  0]
   [  0]
   [  0]
   ...
  ]]]
```

上面的尺寸12，是因为我们之前设定了一个12的batchsize，通道为1是因为我们的数据只有一个通道。**请注意，这一点跟pytorch不同，pytorch是通道位于长宽前面，当然这影响不大，因为如果你稍有不慎写错了你的网络会出问题，最终你是能够发现的**.

接下来要来一点复杂点的操作了，接下来我们要构造一个自己的dataset类，它属于`tf.data.Dataset`类型，它返回的一张图片，以及图片里面目标的boundingbox的array（多个目标有多个box，每个box是4个点坐标）。这为我们后面输入到yolov3模型之中做准备。我们有两种方式来做：

- 用tfrecord，实现提取数据保存到tfrecord中，但我认为这种操作方式比较麻烦；
- 直接在线的从本地读取数据，边训练边读，这也是pytorch的运作方式

尽管我们有这些方式可以选择，但是我们需要选择一种最理智的方式，也就是一劳永逸的方式，采用第一种应该是最好的。这样我们以后所有的检测类型数据集，都可以使用一个统一化的tfrecord接口来读取，而你要做的，仅仅只是写一个脚本将检测数据标注转化为tfrecord文件。后面就只需要读取这个tfrecord文件即可。

那么思路就出来了，我们需要先写一个转换器，把coco转化成tfrecord！首先让我们来实际一下，应该在tfrecord里面存储什么内容：

```python
IMAGE_FEATURE_MAP = {
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/source_id': tf.io.FixedLenFeature([], tf.string),
    'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    'image/object/difficult': tf.io.VarLenFeature(tf.int64),
    'image/object/truncated': tf.io.VarLenFeature(tf.int64),
    'image/object/view': tf.io.VarLenFeature(tf.string),
}
```

我们将一张图片存入为一个`tf.train.Example`的格式，那么我们就可以从tfrecord里面load我们想要的内容。为了编写一个前后端分离的数据load器，我们先来看一下如何写一个转换器，将coco转换成tfrecord。

#### coco转tfrecord

假设我们将coco数据加载，封装为一个字典，比如下面这样的：

```python
img_data = {
  'pixel_data': bytes,
  'bboxes': [[0.2, .3, .4, .5], [.1, .3, .4, .5]],
  'labels': [23, 34]
}
```

在这里，需要记住一条理念：**tfrecord** 将所有的标注看做是一类，什么意思呢？上面的所有的xmin是一类，ymin是一类，label是一类，这也就是说，它虽然是一个整体，但是我们将其看做一类，这样其实可以大大的节省存储空间以及提高数据读取效率。

拿到了一个img_data之后，我们只需要转成 `tf.train.Example`：

```python
def dict_to_coco_example(img_data):
    """Convert python dictionary formath data of one image to tf.Example proto.
    Args:
        img_data: infomation of one image, inclue bounding box, labels of bounding box,\
            height, width, encoded pixel data.
    Returns:
        example: The converted tf.Example
    """
    bboxes = img_data['bboxes']
    xmin, xmax, ymin, ymax = [], [], [], []
    for bbox in bboxes:
        xmin.append(bbox[0])
        xmax.append(bbox[0] + bbox[2])
        ymin.append(bbox[1])
        ymax.append(bbox[1] + bbox[3])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(img_data['height']),
        'image/width': dataset_util.int64_feature(img_data['width']),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/label': dataset_util.int64_list_feature(img_data['labels']),
        'image/encoded': dataset_util.bytes_feature(img_data['pixel_data']),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf-8')),
    }))
    return example
```

OK，关于这个转换器，其实你可以轻而易举的从github获取，本仓库包含了一个转换器，在tools里面，地址：http://github.com/jinfagang/yolov3_tf2 .

事实上，关于tfrecord转换被很多人吐槽，同时它也存在很多弊端，它的缺点十分明显：

- 且不说转换麻烦（得写两次脚本），占用的存储空间极大，coco转完可能得有40-60多个G，总的数据集都没有这么大；
- check数据比较麻烦。

本教程以及附带仓库实现的转换脚本绝非事网上传播的版本，很多人说用别人的代码不好，也有人说什么东西都得自己原创，我觉得这是两个极端，很多东西都原创你会浪费很多时间，但是如果什么东西都照搬你会进入别人的坑。就比如你从网上找一个coco转tfrecord的脚本你至少会遇到两个问题：1. 没有兼容2.0 api，很多接口2.0事没有的；2. 你根本无法用，我在写这篇文章的时候就参考了一些实现，有一个看起来来官方的教程把所有的coco数据读入内存然后再一个一个存。这得多大的内存才不会被系统进程杀死？感兴趣或者不服的朋友可以拿开源的自己去试一试，能使用算我输。

但是不管怎样，转换为tfrecord却可以极大地提高运行的效率。不过作为一个不喜欢全家桶的，我并不喜欢tensorflow的所有生态。我更喜欢简单的一点的东西。那么我们就写一个及其简单的数据load器吧，我们使用 `tf.data.Dataset.from_data_slice` 这个接口来load数据：

```
挖坑 还有点问题
```

#### coco从tfrecord读取

继续秉承有始有终的理念，既然我们打算用tfrecord存了，那就得用tfrecord读取，上来这条贼船，你就是贼了。。

