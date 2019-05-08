import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3.models import YoloV3, YoloV3Tiny
from yolov3.dataset import transform_images
from yolov3.utils import draw_outputs
from google.protobuf import text_format
from alfred.protos.labelmap_pb2 import LabelMap


flags.DEFINE_string('classes', './data/coco.prototxt', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3_coco.ckpt',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')


def get_coco_names(proto_f):
    with open(proto_f, 'r') as f:
        lm = LabelMap()
        lm = text_format.Merge(str(f.read()), lm)
        names_list = [i.display_name for i in lm.item]
        return names_list


def main(_argv):
    if FLAGS.tiny:
        yolo = YoloV3Tiny()
    else:
        yolo = YoloV3()

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = get_coco_names(FLAGS.classes)
    logging.info('classes loaded')

    img = tf.image.decode_image(open(FLAGS.image, 'rb').read(), channels=3)
    img = tf.expand_dims(img, 0)
    img = transform_images(img, FLAGS.size)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))

    img = cv2.imread(FLAGS.image)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(FLAGS.output, img)
    logging.info('output saved to: {}'.format(FLAGS.output))

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
