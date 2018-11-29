import tensorflow as tf
import matplotlib.pyplot as pyplot
import numpy as np

(X_train, y_train_fine), (X_test, y_test_fine) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
def convert_rgb_to_gray(images):
  return (0.2125 * images[:,:,:,:1]) + (0.7154 * images[:,:,:,1:2]) + (0.0721 * images[:,:,:,-1:])

gray_X_train = np.squeeze(convert_rgb_to_gray(X_train), axis=-1)
print(gray_X_train.shape)

# create a grid of 3x6 images
for i in range(0, 9, 2):
  pyplot.subplot(3,6, 1 + i)
  pyplot.imshow(X_train[i].astype('uint32'))
  pyplot.subplot(3,6, 1 + i + 1)
  pyplot.imshow(gray_X_train[i].astype('uint32'), cmap='gray')
# show the plot
pyplot.show()
