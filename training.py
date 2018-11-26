import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras import backend as K
from keras.utils import to_categorical
from keras import layers, models
from layers.coupled_capsule import CoupledConvCapsule
import numpy as np

# 1. Standardize images across the dataset, mean=0, stdev=1
'''standardize pixel values across the entire dataset'''

# K.set_image_dim_ordering('th')
# load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
fine_classes = 100
# reshape to be [samples][pixels][width][height]
# X_train = X_train.reshape(X_train.shape[0], 3, 32, 32)
# X_test = X_test.reshape(X_test.shape[0], 3, 32, 32)
# convert from int to float
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
print('X shape', X_train.shape, X_test.shape)

y_train = to_categorical(y_train, num_classes=fine_classes)	
y_test = to_categorical(y_test, num_classes=fine_classes)
print('y shape', y_train.shape, y_test.shape)
# define data preparation
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
# fit centering and normalization from data
#datagen.fit(X_train)

'''Point of Comparison for Image Augmentation'''
# configure batch size and retrieve one batch of images
'''for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
	print(X_batch[0])
	# create a grid of 3x3 images
	for i in range(0, 9):
		pyplot.subplot(330 + 1 + i)
		pyplot.imshow(X_batch[i].astype(np.int32), cmap=pyplot.get_cmap('brg'), interpolation='nearest')
	# show the plot
	pyplot.show()
	break
'''