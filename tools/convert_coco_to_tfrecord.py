"""

convert coco data to tfrecord 

this file can also changed into converting VOC to tfrecord format.

I will update it into comptable to both VOC and coco dataset

"""

from pycocotools.coco import COCO
from PIL import Image
from random import shuffle
import os, sys
import numpy as np
import tensorflow as tf

from loguru import logger as logging
from alfred.utils.log import init_logger

init_logger()


def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
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
                            labels.append(ann['category_id'])
                        # read image data we need
                        img_path = os.path.join(imgs_dir, img_detail['file_name'])
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
                    logging.error('{} {} does not exist, passing it.'.format(s, year))

    else:
        logging.error('{} not supported yet.'.format(dataset))


def data_to_tf_example(img_data):
    bboxes = img_data['bboxes']
    xmin, xmax, ymin, ymax = [], [], [], []
    for bbox in bboxes:
        xmin.append(bbox[0])
        xmax.append(bbox[0] + bbox[2])
        ymin.append(bbox[1])
        ymax.append(bbox[1] + bbox[3])
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'image/height': int64_feature(img_data['height']),
            'image/width': int64_feature(img_data['width']),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/class/label': int64_feature(img_data['labels']),
            'image/encoded': bytes_feature(img_data['pixel_data']),
            'image/format': bytes_feature('jpeg'.encode('utf-8')),
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
            # for idx, img_data in enumerate(subset_data):
            #     if idx % 100 == 0:
            #         logging.info('Converting images: %d/%d' %
            #                      (idx, len(subset_data)))
            example = data_to_tf_example(data)
            tfrecord_writer.write(example.SerializeToString())
            if i % 1000 == 0:
                logging.info('saved %d examples into tfrecord.' % i)
            i += 1
    logging.info('coco tfrecord generate done.')


if __name__ == "__main__":
    convert_coco(sys.argv[1])
