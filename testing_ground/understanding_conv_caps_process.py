# trying to understand https://github.com/lalonderodney/SegCaps/blob/master/capsule_layers.py#L124-L156
import tensorflow as tf
import numpy as np
import keras.backend as K
from os import environ

# disable GPU
environ['CUDA_VISIBLE_DEVICES'] = ''

tf.enable_eager_execution()
batch = 3
# think of capsules as channels
height, width, capsules, atoms = (512, 512, 4, 16)
kernel_size = 5
stride = 2

ints = np.random.randint(512, size=[batch, height, width, capsules, atoms])
ints = tf.cast(ints, tf.float32)
# reshaped = tf.reshape(ints, [batch * capsules, height, width, atoms])
# print(reshaped.shape)

# if multiple input (batches) are thought to be multiple stacks of input then
# what this does is flips things upwards and realigning as needed
transposed = tf.transpose(ints, [3, 0, 1, 2, 4])
transposed_shape = transposed.shape
# this stacks everything on top of each other, as if each channel of the capsule
# for each input in batch makes up a new input
# -> xxxxx (capsules for each input)
# -> x (stacked like this for each input)
# -> x
# -> x
# -> x
reshaped_two = tf.reshape(transposed, [batch * capsules, height, width, atoms])
print(reshaped_two.shape)
# please check out https://pgaleone.eu/tensorflow/2018/07/28/understanding-tensorflow-tensors-shape-static-dynamic/
# to understand set_shape more. Batch size is unimportant to know but height, width, atoms is needed
reshaped_two.set_shape((None, height, width, atoms))
print(reshaped_two.get_shape())

# in general not true that they will be equal, especially when capsules > 1
# print(np.equal(reshaped, reshaped_two))

desired_capsules = 2
desired_atoms = 16
random_weights = tf.convert_to_tensor(
  np.random.rand(kernel_size, kernel_size, atoms, desired_capsules * desired_atoms),
  dtype=tf.float32
)
convolution = K.conv2d(reshaped_two, random_weights, strides=(2,2), padding='same', data_format='channels_last')
convolution_shape = K.shape(convolution)
print('convolution shape result', convolution_shape)
_, convol_height, convol_width, _ = convolution.get_shape()
print(convol_height, convol_width, convolution.get_shape())
votes = K.reshape(convolution, [batch, capsules, convolution_shape[1], convolution_shape[2], desired_capsules, desired_atoms])
votes.set_shape((None, capsules, convol_height.value, convol_width.value, desired_capsules, desired_atoms))
print('shape of votes (result from convolution)', votes.shape)
logit_shape = K.stack([batch, capsules, convolution_shape[1], convolution_shape[2], desired_capsules])
logits = tf.fill(logit_shape, 0.0)
# shape is batch, capsules, convol height, convol width, desired_capsules
print('shape of logits', logits.shape)
route = tf.nn.softmax(logits, dim=-1)
print('Shape of routes', route.shape)
# print(route)
# make first dimension be the atoms
# shape will be atoms, batch, capsules, convole height, convole width, desired_capsules
transpose_votes = tf.transpose(votes, [5, 0, 1, 2, 3, 4])
print('Shape of transposed votes', transpose_votes.shape)
# taking advantage of broadcast
unrolled_unsquashed_activation = route * transpose_votes
print('shape of unsquashed activation (before summation and desired atoms still the first axis)', unrolled_unsquashed_activation.shape)
# transpose back 
unsquashed_activation = tf.transpose(unrolled_unsquashed_activation, [1,2,3,4,5,0])
# shape will be [batch, channels, convol height, convol width, desired channels, desired atoms] 
print('shape of unsquashed activation (before summation and after transpose)', unsquashed_activation.shape)

# sum over the channels (this is the p_xy in Capsules for Object Segmentation paper) 
unsquashed_activation = tf.reduce_sum(unsquashed_activation, axis=1)
print('shape of unsquashed activation (after summation)', unsquashed_activation.shape)

# and the rest is squashing, calculating logits (similar to activation above), and getting the final activation
# the result will be a tensor shaped batch, convol height, convol width, desired channels, desired atoms