import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras import backend as K
import numpy as np

tf.enable_eager_execution()

# 1. Standardize images across the dataset, mean=0, stdev=1
'''standardize pixel values across the entire dataset'''
# 2. ZCA whitining REF http://ufldl.stanford.edu/wiki/index.php/Whitening 
'''When using ZCA whitening (unlike PCA whitening), 
we usually keep all \textstyle n dimensions of the data, and do not try to reduce its dimension. '''

K.set_image_dim_ordering('th')
# load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 3, 32, 32)
X_test = X_test.reshape(X_test.shape[0], 3, 32, 32)
# convert from int to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# define data preparation
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, zca_whitening=True)
# fit parameters from data
datagen.fit(X_train)

'''Point of Comparison for Image Augmentation'''
# configure batch size and retrieve one batch of images
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
	# create a grid of 3x3 images
	for i in range(0, 9):
		pyplot.subplot(330 + 1 + i)
		pyplot.imshow(X_batch[i].reshape(32, 32, 3), cmap=pyplot.get_cmap('brg'))
	# show the plot
	pyplot.show()
	break




