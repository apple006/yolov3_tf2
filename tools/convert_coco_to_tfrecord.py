"""

convert coco data to tfrecord 

this file can also changed into converting VOC to tfrecord format.

I will update it into comptable to both VOC and coco dataset

"""

from pycocotools.coco import COCO
from PIL import Image
from random import shuffle
import os
import sys
import numpy as np
import tensorflow as tf
from alfred.utils.log import logger as logging


coco_remap_dict = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39,
                   46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79}


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def load_data(dataset='coco', root_dir=''):
    """
    this function will support coco and VOC
    """
    if dataset == 'coco':
        for s in ['train', 'val']:
            for year in ['2014', '2017']:
                # every set, year will be yeild
                # data = []
                imgs_dir = os.path.join(root_dir, '{}{}'.format(s, year))
                anno_f = os.path.join(root_dir, 'annotations',
                                      'instances_{}{}.json'.format(s, year))
                if os.path.exists(imgs_dir) and os.path.exists(anno_f):
                    logging.info('solving COCO: {} {}'.format(s, year))
                    coco = COCO(anno_f)
                    # totally 82783 images
                    img_ids = coco.getImgIds()
                    # 90 categories (not continues, actually only has 80)
                    cat_ids = coco.getCatIds()

                    for idx, img_id in enumerate(img_ids):
                        if idx % 500 == 0:
                            logging.info('Reading images: %d/%d' %
                                         (idx, len(img_ids)))
                        img_info = dict()
                        bboxes = []
                        labels = []

                        img_detail = coco.loadImgs(img_id)[0]
                        h = img_detail['height']
                        w = img_detail['width']

                        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
                        anns = coco.loadAnns(ann_ids)
                        for ann in anns:
                            bboxes_data = ann['bbox']
                            # normalize box
                            bboxes_data = [
                                bboxes_data[0] / float(w),
                                bboxes_data[1] / float(h),
                                bboxes_data[2] / float(w),
                                bboxes_data[3] / float(h),
                            ]
                            bboxes.append(bboxes_data)
                            # this category_id should be remap
                            labels.append(coco_remap_dict[int(ann['category_id'])])
                        # read image data we need
                        img_path = os.path.join(
                            imgs_dir, img_detail['file_name'])
                        img_bytes = open(img_path, 'rb').read()

                        img_info['pixel_data'] = img_bytes
                        img_info['height'] = h
                        img_info['width'] = w
                        img_info['bboxes'] = bboxes
                        img_info['labels'] = labels
                        yield img_info
                        # data.append(img_info)
                    # yield data
                else:
                    logging.error(
                        '{} {} does not exist, passing it.'.format(s, year))
                    logging.error(
                        '{} and {} not exist.'.format(imgs_dir, anno_f))

    else:
        # TODO: adding VOC, KITTI, converting
        logging.error('{} not supported yet.'.format(dataset))


def data_to_tf_example(img_data):
    bboxes = img_data['bboxes']
    xmin, xmax, ymin, ymax = [], [], [], []
    for _, bbox in enumerate(bboxes):
        xmin.append(bbox[0])
        xmax.append(bbox[0] + bbox[2])
        ymin.append(bbox[1])
        ymax.append(bbox[1] + bbox[3])
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'image/height': _int64_feature(img_data['height']),
            'image/width': _int64_feature(img_data['width']),
            'image/object/bbox/xmin': _float_feature(xmin),
            'image/object/bbox/xmax': _float_feature(xmax),
            'image/object/bbox/ymin': _float_feature(ymin),
            'image/object/bbox/ymax': _float_feature(ymax),
            # using class id directly, not using label text
            'image/object/class/label': _int64_feature(img_data['labels']),
            'image/encoded': _bytes_feature(img_data['pixel_data']),
            'image/format': _bytes_feature('jpeg'.encode('utf-8')),
        }))
    return example


def convert_coco(root_dir):
    logging.info('start to converting coco tfrecord...')
    data_iter = load_data(dataset='coco', root_dir=root_dir)
    tfrecord_f = os.path.join(root_dir, 'coco_trainval_20142017.tfrecord')
    logging.info('saving tfrecord to file: {}'.format(tfrecord_f))

    with tf.io.TFRecordWriter(tfrecord_f) as tfrecord_writer:
        i = 0
        for data in data_iter:
            try:
                example = data_to_tf_example(data)
                tfrecord_writer.write(example.SerializeToString())
                if i % 1000 == 0:
                    logging.info('saved %d examples into tfrecord.' % i)
                i += 1
            except KeyboardInterrupt:
                logging.info('interrupted... try exit savely..')
                tfrecord_writer.close()
                logging.info('Currently converted data saved.')
    logging.info('coco tfrecord generate done.')


if __name__ == "__main__":
    convert_coco(sys.argv[1])
