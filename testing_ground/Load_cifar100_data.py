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
print(y_train[0:5])
# Btara's comment: some reminders
# 1. Want to normalize input and seems ImageDataGenerator can help us with that https://keras.io/preprocessing/image/#imagedatagenerator-class. Also check out https://stackoverflow.com/questions/41855512/how-does-data-normalization-work-in-keras-during-prediction for ideas on how to use it
# 2. Remember to turn the labels to categorical, Keras has a utility function for that it seems https://keras.io/utils/#to_categorical 
# 3. Remember that in the end we want to pass batches of image data to either model.fit or model.fit_generator (see further below for links to model guide and and also look at Rodney's code)

print(x_train.shape)
#train = data[0][:] #

img = array_to_img(x_train[49999]) # image 32 X 32 dimention ==> 32 lists, each list with 32 lists of 3 RGB values
img.show() # image with label 73

#new_training_list, validation_list = train_test_split(train, test_size=0.1, random_state=7)
#x_train1=tf.convert_to_tensor(x_train)#dtype=tf.float32)

x_train1 = tf.cast(x_train, tf.float32)
x_train1.shape

# Btara's comments: The transpose is not needed for normal conv2D
#transposed = tf.transpose(x_train1, [3,0, 1, 2])#         batch, , channel , capsules, atoms
#transposed_shape = transposed.shape

# In Keras (and basically tensorflow) there is the convolution function and the convolution as a layer. Since am assuming here
# you want to see the convolution result, then what you'd want is keras.backend.pool2D instead: see https://keras.io/backend/
#conv1 =  layers.Conv2D(filters =16, kernel_size = 5, strides = 2 ,  padding = 'same', activation ='relu', name ='conv1'  ) (transposed)
conv = K.conv2d(x_train1, kernel=tf.constant(np.random.rand(5,5,3,4), dtype=tf.float32), strides=(2,2), padding='same')
print(conv.shape)

# Btara's comment: In order to build layers you would need to start from the Input layer i.e an Input(shape=(32,32,3)) for the Cifar100 dataset
# then we compond the layer just like in https://github.com/lalonderodney/SegCaps/blob/master/capsnet.py#L17 and as detailed in this guide
# https://keras.io/getting-started/functional-api-guide/

# Note that the tensor returned by Input is special (unless tensor argument is given, but we don't need to). It is what is known in tensorflow as
# a placeholder: https://www.tensorflow.org/api_docs/python/tf/placeholder. It's basically tensor to be filled in with data in the future:w

# Btara's comments: so here this part is not needed
# we'd probably want to do a normal 2D convolution and then treat the result of the convolution as a
# capsule layer, just like in https://github.com/lalonderodney/SegCaps/blob/master/capsnet.py#L25.
# We do so by reshaping (through a Reshape layer) after the Conv2D
# Note: Dimensions specified in Keras layers do not include the batch size (so you don't need to care about the 50k or other batch size in the layers)
reshaped_two = tf.reshape(x_train1, [50000, 16, atoms,capsules,channels])# 4 capsules and 16 atoms 
print(reshaped_two.shape)
reshaped_two.set_shape((None, 16, 16, 4,3))



