import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from absl import app, flags
from absl.flags import FLAGS
from alfred.utils.log import logger as logging

from yolov3.models import *
from yolov3.utils import freeze_all
import dataset.dataset_coco as dataset


flags.DEFINE_string('dataset', '/media/jintain/wd/permenant/datasets/coco/*.tfrecord',
                    'dataset tfrecor path')
flags.DEFINE_string('val_dataset', '', 'path to validation dataset')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
# actually, both model architecture and weights will be saved
flags.DEFINE_string('weights', './checkpoints/yolov3_coco-{}.ckpt',
                    'path to weights file')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_enum(
    'mode', 'eager_tf', ['fit', 'eager_fit', 'eager_tf'], 'fit: model.fit, '
    'eager_fit: model.fit(run_eagerly=True), '
    'eager_tf: custom GradientTape')
flags.DEFINE_enum(
    'transfer', 'none',
    ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
    'none: Training from scratch, '
    'darknet: Transfer darknet, '
    'no_output: Transfer all but output, '
    'frozen: Transfer and freeze all, '
    'fine_tune: Transfer all and freeze darknet only')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('epochs', 102, 'number of epochs')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_integer('val_per_epoch', 32, 'val_per_epoch')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_boolean('resume', True, 'resume from checkpoints/, if model not exist, not resume')


def main(_argv):
    if FLAGS.tiny:
        logging.info('using YoloV3 Tiny model.')
        model = YoloV3Tiny(FLAGS.size, training=True)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        logging.info('using YoloV3 model.')
        model = YoloV3(FLAGS.size, training=True)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    train_dataset = dataset.load_fake_dataset()
    if FLAGS.dataset and os.path.exists(os.path.dirname(FLAGS.dataset)):
        logging.info(f'loading dataset from: {FLAGS.dataset}')
        train_dataset = dataset.load_tfrecord_dataset(FLAGS.dataset,
                                                      FLAGS.classes)
    else:
        logging.info('{} can not found, did you changed to your machine path?'.format(FLAGS.dataset))
        exit(0)
    train_dataset = train_dataset.shuffle(buffer_size=1024)  # TODO: not 1024
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.map(lambda x, y: (dataset.transform_images(
        x, FLAGS.size), dataset.transform_targets(y, anchors, anchor_masks, 80)
    ))
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = dataset.load_fake_dataset()
    if FLAGS.val_dataset:
        val_dataset = dataset.load_tfrecord_dataset(FLAGS.val_dataset,
                                                    FLAGS.classes)
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(lambda x, y: (dataset.transform_images(
        x, FLAGS.size), dataset.transform_targets(y, anchors, anchor_masks, 80)
    ))

    if FLAGS.transfer != 'none':
        if FLAGS.transfer == 'fine_tune':
            darknet = model.get_layer('yolo_darknet')
            freeze_all(darknet)
        elif FLAGS.mode == 'frozen':
            freeze_all(model)
        else:
            # reset top layers
            if FLAGS.tiny:  # get initial weights
                init_model = YoloV3Tiny(FLAGS.size, training=True)
            else:
                init_model = YoloV3(FLAGS.size, training=True)
            if FLAGS.transfer == 'darknet':
                for l in model.layers:
                    if l.name != 'yolo_darknet' and l.name.startswith('yolo_'):
                        l.set_weights(
                            init_model.get_layer(l.name).get_weights())
                    else:
                        freeze_all(l)
            elif FLAGS.transfer == 'no_output':
                for l in model.layers:
                    if l.name.startswith('yolo_output'):
                        l.set_weights(
                            init_model.get_layer(l.name).get_weights())
                    else:
                        freeze_all(l)
    start_epoch = 1
    if FLAGS.resume:
        latest_cp = tf.train.latest_checkpoint(os.path.dirname(FLAGS.weights))
        if latest_cp:
            start_epoch = int(latest_cp.split('-')[1].split('.')[0])
            model.load_weights(latest_cp)
            logging.info('model resumed from: {}, start at epoch: {}'.format(latest_cp, start_epoch))
        else:
            logging.info('passing resume since weights not there. training from scratch')

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [YoloLoss(anchors[mask]) for mask in anchor_masks]

    if FLAGS.mode == 'eager_tf':
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        for epoch in range(start_epoch, FLAGS.epochs + 1):
            for batch, (images, labels) in enumerate(train_dataset):
                try:
                    with tf.GradientTape() as tape:
                        outputs = model(images, training=True)
                        regularization_loss = tf.reduce_sum(model.losses)
                        pred_loss = []
                        for output, label, loss_fn in zip(outputs, labels, loss):
                            pred_loss.append(loss_fn(label, output))
                        total_loss = tf.reduce_sum(pred_loss) + regularization_loss
                    grads = tape.gradient(total_loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    avg_loss.update_state(total_loss)
                    
                    if batch % 10 == 0:
                        logging.info("Epoch: {}, iter: {}, total_loss: {:.4f}, pred_loss: {}".format(
                        epoch, batch, total_loss.numpy(), list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                    if batch % 500 == 0 and batch != 0:
                        logging.info('save model periodically...')
                        model.save_weights(FLAGS.weights.format(epoch))
                except KeyboardInterrupt:
                    logging.info('interrupted. try saving model now...')
                    model.save_weights(FLAGS.weights.format(epoch))
                    logging.info('model has been saved.')
                    exit(0)
                except Exception as e:
                    logging.info('got an unexpected error: {}, continue...'.format(e))
                    continue
            if epoch % FLAGS.val_per_epoch == 0 and epoch != 0:
                for batch, (images, labels) in enumerate(val_dataset):
                    outputs = model(images)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                    logging.info("{}_val_{}, {}, {}".format(
                        epoch, batch, total_loss.numpy(),
                        list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                    avg_val_loss.update_state(total_loss)
                logging.info("{}, train: {}, val: {}".format(
                    epoch,
                    avg_loss.result().numpy(),
                    avg_val_loss.result().numpy()))

                avg_loss.reset_states()
                avg_val_loss.reset_states()
        model.save_weights(FLAGS.weights.format(epoch))
        # save final model (both arcchitecture and weights)
        model.save('yolov2_coco.h5')
        logging.info('training done.')
        exit(0)
    else:
        model.compile(optimizer=optimizer,
                      loss=loss,
                      run_eagerly=(FLAGS.mode == 'eager_fit'))
        callbacks = [
            ReduceLROnPlateau(verbose=1),
            EarlyStopping(patience=3, verbose=1),
            ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
                            verbose=1,
                            save_weights_only=True),
            TensorBoard(log_dir='logs')
        ]
        history = model.fit(train_dataset,
                            epochs=FLAGS.epochs,
                            callbacks=callbacks,
                            validation_data=val_dataset)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
