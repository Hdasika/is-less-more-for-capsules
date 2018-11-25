import tensorflow as tf 
import keras.backend as K
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, array_to_img 
from keras import layers
#from tensorflow.keras.preprocessing.image import 
#from sklearn.model_selection import train_test_split


tf.enable_eager_execution()
channels = 3
kernel_size = 5
stride = 2

height, width , capsules , atoms =(32, 32,  4, 16)


#Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test)
data= tf.keras.datasets.cifar100.load_data(label_mode='fine') # load cifar100 with 100 sub class labels (0-99)

x_train = data[0][0] # 50K  images and each row is one image
y_train = data[0][1] # lables 

#train = data[0][:] #

img = array_to_img(x_train[49999]) # image 32 X 32 dimention ==> 32 lists, each list with 32 lists of 3 RGB values
img.show() # image with label 73

#new_training_list, validation_list = train_test_split(train, test_size=0.1, random_state=7)
#x_train1=tf.convert_to_tensor(x_train)#dtype=tf.float32)

x_train1 = tf.cast(x_train, tf.float32)
x_train1.shape

#transposed = tf.transpose(x_train1, [3,0, 1, 2])#         batch, , channel , capsules, atoms
#transposed_shape = transposed.shape
#conv1 =  layers.Conv2D(filters =16, kernel_size = 5, strides = 2 ,  padding = 'same', activation ='relu', name ='conv1'  ) (transposed)

reshaped_two = tf.reshape(x_train1, [50000, 16, atoms,capsules,channels])# 4 capsules and 16 atoms 
reshaped_two.shape
reshaped_two.set_shape((None, 16, 16, 4,3))



