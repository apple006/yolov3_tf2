"""

dataset loader of coco
using tensorflow 2.0 recommended api

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


def input_map_fn(img, label):
    pass


def load_coco_2014_dataset(root_dir, batch_size=12):
    imgs_dir = os.path.join(root_dir, 'train2014')
    anno_f = os.path.join(root_dir, 'annotations', 'instance_train2014.json')
    coco = COCO(anno_f)

    # totally 82783 images
    img_ids = coco.getImgIds()
    # 90 categories (not continues, actually only has 80)
    cat_ids = coco.getCatIds()

    all_imgs = []
    all_labels = []
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

        img_path = os.path.join(root_dir, img_detail['file_name'])
        all_imgs.append(img_path)
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
        all_labels.append(bboxes)
    train_ds = tf.data.Dataset().from_tensor_slices(
        (tf.constant(all_images), 
        tf.constant(all_labels))
        ).shuffle(buffer_size=10000).map(input_map_fn).batch(batch_size)
    return train_ds


if __name__ == "__main__":
    train_ds = load_coco_2014_dataset()
    for img, label in train_ds.take(1):
        print(img)
        print(label)

        # start training on model...
        break