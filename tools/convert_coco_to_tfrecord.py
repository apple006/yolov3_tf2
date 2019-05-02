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


def load_data(dataset='coco', root_dir):
    """
    this function will support coco and VOC
    """
    if dataset == 'coco':
        for s in ['train', 'val']:
            for year in ['2014', '2017']:
                # every set, year will be yeild
                data = dict()
                imgs_dir = os.path.join(root_dir, '{}{}'.format(s, year))
                anno_f = os.path.join(root_dir, 'annotations', 'instances_{}{}'.format(s, year))
                logging.info('solving COCO: {} {}'.format(s, year))
                coco = COCO(anno_f)
                # totally 82783 images
                img_ids = coco.getImgIds()
                # 90 categories (not continues, actually only has 80)
                cat_ids = coco.getCatIds()

                for idx, img_id in enumerate(img_ids):
                    if idx % 500 == 0:
                        logging.info('Reading images: %d/%d'%(idx, len(img_ids)))
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
                            bboxes_data[0]/float(w),
                            bboxes_data[1]/float(h),
                            bboxes_data[2]/float(w),
                            bboxes_data[3]/float(h),
                        ]
                        bboxes.append(bboxes_data)
                        labels.append(ann['category_id'])
                    # read image data we need
                    img_path = os.path.join(root_dir, img_detail['file_name'])
                    img_bytes = tf.gfile.FastGFile(img_path, 'rb').read()

                    img_info['pixel_data'] = img_bytes
                    img_info['height'] = h
                    img_info['width'] = w
                    img_info['bboxes'] = bboxes
                    img_info['labels'] = labels
                    data.append(img_info)
                yield data
    else:
        logging.error('{} not supported yet.'.format(dataset))


def data_to_tf_example(img_data):
    bboxes = img_data['bboxes']
    xmin, ymax, ymin, ymax = [], [], [], []
    for bbox in bboxes:
        xmin.append(bbox[0])
        xmax.append(bbox[0] + bbox[2])
        ymin.append(box[1])
        ymax.append(bbox[1] + bbox[3])
    
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
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


def convert_coco(root_dir):
    logging.info('start to converting coco tfrecord...')
    data_iter = load_img_data(dataset='coco', root_dir=root_dir)
    with tf.python_io.TFRecordWriter('coco_trainval_20142017.tfrecord') as tfrecord_writer:
        for subset_data in data_iter:
            for idx, img_data enumerate(subset_data):
                if idx % 100 == 0:
                    logging.info('Converting images: %d/%d'%(idx, len(subset_data)))
                example = data_to_tf_example(img_data)
                tfrecord_writer.write(example.SerializeToString())
    logging.info('coco tfrecord generate done.')

    

