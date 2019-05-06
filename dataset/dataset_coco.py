"""

dataset loader of coco
using tensorflow 2.0 recommended api

"""
from pycocotools.coco import COCO
from PIL import Image
from random import shuffle
import os
import sys
import numpy as np
import tensorflow as tf
import cv2
from loguru import logger as logging
from alfred.utils.log import init_logger

init_logger()
this_dir = os.path.dirname(os.path.abspath(__file__))


IMAGE_FEATURE_MAP = {
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
}


def parse_tfrecord(tfrecord, class_table):
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    x_train = tf.image.resize(x_train, (416, 416))
    # get numpy from x_train
    label_idx = x['image/object/class/label']
    labels = tf.sparse.to_dense(label_idx)
    labels = tf.cast(labels, tf.float32)
    y_train = tf.stack([
        tf.sparse.to_dense(x['image/object/bbox/xmin']),
        tf.sparse.to_dense(x['image/object/bbox/ymin']),
        tf.sparse.to_dense(x['image/object/bbox/xmax']),
        tf.sparse.to_dense(x['image/object/bbox/ymax']), 
        labels
    ], axis=1)
    paddings = [[0, 100 - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)
    return x_train, y_train


def load_tfrecord_dataset(file_pattern, class_file):
    LINE_NUMBER = -1  # TODO: use tf.lookup.TextFileIndex.LINE_NUMBER
    class_table = tf.lookup.StaticHashTable(
        tf.lookup.TextFileInitializer(class_file,
                                      tf.string,
                                      0,
                                      tf.int64,
                                      LINE_NUMBER,
                                      delimiter="\n"), -1)
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    return dataset.map(lambda x: parse_tfrecord(x, class_table))


if __name__ == "__main__":
    train_ds = load_tfrecord_dataset(sys.argv[1],
                                     os.path.join(this_dir, 'coco.names'))
    logging.info('dataset reading complete.')
    for img, label in train_ds.take(1):
        print(img.numpy())
        print(label.numpy())
        # start training on model...
        img = np.array(img.numpy(), np.uint8)
        cv2.imshow('rr', img)
        cv2.waitKey(0)
        break
