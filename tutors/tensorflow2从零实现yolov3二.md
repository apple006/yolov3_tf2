# TensorFlow2.0 从零实现YoloV3检测网络之TF2.0训练MobileNetV3

> 在正式开始教程之前，需要强调一下，**这不仅仅是一篇教你从零实现一个yolov3检测器的教程，同时也是一个最新最详尽比较权威中肯的TensorFlow2.0教程（我们会包含从dataloader到基础keras api网络搭建的所有过程）**. 同时欢迎大家来我们的论坛探讨AI问题: ft.manaai.cn, 同时也欢迎大家支持一下我们做的AI市场平台，海量AI算法开箱即用：http://manaai.cn .


接着上一讲，继续为您说. 最近关于MobileNetV3的论文可以说是非常的非常的火热，可见大家对于移动端或者高性能的基础网络架构是多么的关注。那么，秉着最快最全最新最有卵用的原则，本篇教程进行上一个节奏，继续为大家更新。上回我们说道要做一个检测器，从0开始，我们已经经过dataloader部分，也就是从coco转tfrecord，再用tfrecord统一作为我们训练的数据导入器，但是还没有涉及到tensorflow2.0的网络api部分，这篇将给大家讲述这些内容：

- 从零编写MobileNetV3；
- 不仅要编写，还要用一个小的数据集来训练它；
- 不仅要训练它，我们还要测试它相对比MobileNetV2在实际场景下速度的提升。

综上，我们这篇将把主线程稍微的偏离一下，来完成这个现任务先，一来让大家tensorflow2.0 api有所熟悉，为后面做准备，二来给大家用**实际的例子** 展示一下MobilenetV3到底有多牛逼。闲话不多说，先上代码。

## TF2.0编写MobilenetV3网络

关于TF2.0的MobileNetV3我也是参考了一些网络的实现，但是有一点是肯定的，那就是网上没有一个真正从实现到训练的实现，能正确把网络实现的都比较少，纸上得来终觉浅绝知此事要躬行呀！既然如此，本教程就来完成这个光荣而艰巨的使命吧。在编写MobileNetV3的网络之前，我得向大家传授一下在TF2.0中，**神经网络的标准写法**(当然，如果你觉得标准应该属于Google，那可以称之为M式写法），在tf2.0里面，大家千万不要瞎几把乱写，都有套路的。
简单来说，一个标准的Model应该这样写：

```python
import tensorflow as tf

class MobileNetV3Small(tf.keras.Model):
    def __init__(
            self,
            classes: int=1001,
            width_multiplier: float=1.0,
            scope: str="MobileNetV3",
            divisible_by: int=8,
    ):
        super(MobileNetV3Small, self).__init__(name=scope)
    def call(self, input, training=False):
        x = self.yourlayer(input)
        return x
```

核心在于，集成 `tf.keras.Model` 这个类，这样你就可以访问所有Model的方法，比如 `save_weights()`,  `load_weights()`等，而这些都是在tf内部自动完成的。所有从这个角度来说，**tensorflow2.0真的简化了很多操作，并且和pytorch的易用性比肩**.
关于这个网络详细的代码我不贴了，太占篇幅，可以直接从我的repo：http://github.com/jinfagang/tfboys 里面找到相关代码，**同时也包含了pytorch的实现**.


## 使用flowers训练MobileNetV3

本教程的精华部分来了，上一讲大概说了下如何导入数据，但是请相信我，真正麻烦的应该是把数据和网络结合在一起，具体来说你要做 这些事情来训练一个网络：

- 拿到数据，也就是`tf.data.Dataset`;
- 编写你的网络，也就是得到一个 `tf.keras.Model`;
- 开始你的optimizer，你的lossfunction，你的accuracymetric；
- 编写一个循环开始训练。

OK，有了思路就好办了，还记得我在上一篇说的三个网路编写原则吗？

- 你要处理所有情况，包括断点续训、ctrl+c保存权重等；
- 你要打印出清晰的loss信息；

看起来比较复杂，不过好消息是，我们有了tf2.0！！所有的东西你可以选择简单也可以选择复杂。我们先来看看简单的：

```python
 model = tf.keras.applications.MobileNetV2(input_shape=(target_size, target_size, 3), weights=None, include_top=True, classes=5)
   
# todo: why keras fit converge faster than tf loop?
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
try:
    model.fit(
    train_dataset, epochs=50,             
    steps_per_epoch=700,)
except KeyboardInterrupt:
    model.save_weights(ckpt_path.format(epoch=0))
    logging.info('keras model saved.')
model.save_weights(ckpt_path.format(epoch=0))
model.save(os.path.join(os.path.dirname(ckpt_path), 'flowers_mobilenetv2.h5'))

```
就这几行代码，你就可以实现一个分类网络训练的pipeline了。并且可以在你想停止的时候及时的保存权重。

但是你可能会觉得，我还是想自己写循环，来控制每一个epoch的具体运算和操作。没有问题，同样的简单：

```python
loss_fn = tf.losses.SparseCategoricalCrossentropy()
        optimizer = tf.optimizers.RMSprop()

train_loss = tf.metrics.Mean(name='train_loss')
# the accuracy calculation has some problems, seems not right?
train_accuracy = tf.metrics.SparseCategoricalAccuracy(name='train_accuracy')

for epoch in range(start_epoch, 120):
    try:
        for batch, data in enumerate(train_dataset):
            # images, labels = data['image'], data['label']
            images, labels = data
            with tf.GradientTape() as tape:
                predictions = model(images)
                loss = loss_fn(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)
            train_accuracy(labels, predictions)
            if batch % 10 == 0:
                logging.info('Epoch: {}, iter: {}, loss: {}, train_acc: {}'.format(
                epoch, batch, train_loss.result(), train_accuracy.result()))
    except KeyboardInterrupt:
        logging.info('interrupted.')
        model.save_weights(ckpt_path.format(epoch=epoch))
        logging.info('model saved into: {}'.format(ckpt_path.format(epoch=epoch)))
        exit(0)
```

看，这边是TF2.0的核心要义所在。你既可以使用kera的model.fit也可以自己写训练来计算每一个epoch。

下面着重讲解一下tf2.0的训练具体编写不走，别看这几行代码简单，但是实际上跟以往有很大的不同。

1. 首先上来你得准备你好loss function，optimizer，这一点和pytorch很像啊，几乎就是copy啊：
```python
loss_fn = tf.losses.SparseCategoricalCrossentropy()
        optimizer = tf.optimizers.RMSprop()
train_loss = tf.metrics.Mean(name='train_loss')
```
这里的loss_fn就是你的loss计算方式，我们这里采用交叉商，因为是多类别分类问题。后面我们对yolov3进行loss计算就得自己定义loss了。
2. 拿到网络的梯度用`tf.GradientTape()`.
这个操作其实就是把网络前向，然后框架自动求导去更新权重，Tape可以理解为梯度的胶带吧，每一个输出的量都胶了一点梯度：
```python
with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_fn(labels, predictions)

```
上面三行代码基本就是就是核心了。反向传播以前很复杂，现在？就三行核心代码。**请注意，这里就是核心，整片教程其实就这三行代码**。

还记得上一张讲的 `tensorflow_datasets`这个库吗？现在可以用了，我们刚好用它来load一个花的数据集。下面**我们将用100行代码实现一个TF2.0的分类训练代码**.

```python
from alfred.dl.tf.common import mute_tf
mute_tf()
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from alfred.utils.log import logger as logging
import os
import sys

"""
A Simple training pipeline on TensorFlow 2.0 API
with about 60 epochs, model achieves 90% accuracy (training from scratch)
Epoch 1/50
1700/1700 [==============================] - 74s 106ms/step - loss: 0.7693 - accuracy: 0.7104
...
Epoch 44/50
1700/1700 [==============================] - 46s 66ms/step - loss: 0.2280 - accuracy: 0.9182
"""


target_size = 224
use_keras_fit = False
# use_keras_fit = True
ckpt_path = './checkpoints/no_finetune/flowers_mbv2_scratch-{epoch}.ckpt'


def preprocess(x):
    """
    minus mean pixel or normalize?
    """
    x['image'] = tf.image.resize(x['image'], (target_size, target_size))
    x['image'] /= 255.
    x['image'] = 2*x['image'] - 1
    return x['image'], x['label']

def train():
    # using mobilenetv2 classify tf_flowers dataset
    dataset, _ = tfds.load('tf_flowers', with_info=True)
    train_dataset = dataset['train']
    train_dataset = train_dataset.shuffle(100).map(preprocess).batch(4).repeat()

    # init model
    model = tf.keras.applications.MobileNetV2(input_shape=(target_size, target_size, 3), weights=None, include_top=True, classes=5)
    # model.summary()
    # model = tf.keras.models.load_model('flowers_mobilenetv2.h5')
    logging.info('model loaded.')
    
    start_epoch = 0
    latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(ckpt_path))
    if latest_ckpt:
        start_epoch = int(latest_ckpt.split('-')[1].split('.')[0])
        model.load_weights(latest_ckpt)
        logging.info('model resumed from: {}, start at epoch: {}'.format(latest_ckpt, start_epoch))
    else:
        logging.info('passing resume since weights not there. training from scratch')

    if use_keras_fit:
        # todo: why keras fit converge faster than tf loop?
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        try:
            model.fit(
            train_dataset, epochs=50,             
            steps_per_epoch=700,)
        except KeyboardInterrupt:
            model.save_weights(ckpt_path.format(epoch=0))
            logging.info('keras model saved.')
        model.save_weights(ckpt_path.format(epoch=0))
        model.save(os.path.join(os.path.dirname(ckpt_path), 'flowers_mobilenetv2.h5'))
    else:
        loss_fn = tf.losses.SparseCategoricalCrossentropy()
        optimizer = tf.optimizers.RMSprop()

        train_loss = tf.metrics.Mean(name='train_loss')
        # the accuracy calculation has some problems, seems not right?
        train_accuracy = tf.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        for epoch in range(start_epoch, 120):
            try:
                for batch, data in enumerate(train_dataset):
                    # images, labels = data['image'], data['label']
                    images, labels = data
                    with tf.GradientTape() as tape:
                        predictions = model(images)
                        loss = loss_fn(labels, predictions)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    train_loss(loss)
                    train_accuracy(labels, predictions)
                    if batch % 10 == 0:
                        logging.info('Epoch: {}, iter: {}, loss: {}, train_acc: {}'.format(
                        epoch, batch, train_loss.result(), train_accuracy.result()))
            except KeyboardInterrupt:
                logging.info('interrupted.')
                model.save_weights(ckpt_path.format(epoch=epoch))
                logging.info('model saved into: {}'.format(ckpt_path.format(epoch=epoch)))
                exit(0)



if __name__ == "__main__":
    train()

```

如果你找不到`alfred`, 通过 `sudo pip3 install alfred-py`进行安装。这个库得安利一下：

```python
from alfred.dl.tf.common import mute_tf
mute_tf()
```
两行代码，自动屏蔽tensorflow烦人的log。其实很简单，但是很方便。这个库还有很多一些功能，可以去github star一下：http://github.com/jinfagang/alfred


## 小结

在我的repo里面，有训练MobileNetV3的脚本，跟MobileNetV2差不多，我们在这篇博客里面学习了如何从零编写网络，使用tf2.0 api来训练一个图片分类器。
下一篇，我们将继续从0实现YoloV3 in TensorFlow2.0。 并且我会更新一些MobileNetV3和MobileNetV2的实际使用速度对比。（仓库模型我会随时更新，大家可以pull）

