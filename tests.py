import tensorflow as tf
import tensorflow_datasets as tfds


dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']


print(train_dataset)
print(test_dataset)
