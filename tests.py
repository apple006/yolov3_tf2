import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import cv2


dataset, metadata = tfds.load('coco2014', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

train_dataset = train_dataset.shuffle(100).batch(12).repeat()

for img, label in train_dataset.take(1):
    img = img.numpy()
    a = img[0]
    cv2.imshow('rr', a)
    cv2.waitKey(0)
    
