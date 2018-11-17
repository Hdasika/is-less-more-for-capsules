import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

sess = tf.Session()
shaped_randint = np.random.randint(128, size=(4,4,4,2,5))
tf_shaped_randint = tf.convert_to_tensor(shaped_randint, dtype=tf.float32)
normed_tf_shaped_randint = tf.norm(tf_shaped_randint, ord='euclidean', axis=-1)
print(normed_tf_shaped_randint.shape)
max_pooled_with_argmax = tf.nn.max_pool_with_argmax(normed_tf_shaped_randint, ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')
# shape would be 4 x 2 x 2 x 2 (batch x pooled height x pooled width x channels)
max_pooled = max_pooled_with_argmax.output
argmax = max_pooled_with_argmax.argmax
# flatten first to be 128 x 1
argmax = tf.reshape(argmax, shape=[-1])
# unraveled would be 4 x 32 (32 coming from 4 x 2 x 2 x 2, dimension after pooling)
# four refers to the rank of the input that was max pooled over (so actually the height contains information about the full indices)
# it is finding out the correct indices for a particular dims given a flattened index list (lik argmax above)
unraveled = tf.unravel_index(argmax, dims=(4,4,4,2))
print(unraveled.shape)
# transpose, 32 x 4
transposed_unraveled = tf.transpose(unraveled, (1, 0))
print(transposed_unraveled.shape)
# this is picking out 32 elements based on some set of indices which we got from argmax
# result will be 32 x 1
manual_max_pool = tf.gather_nd(normed_tf_shaped_randint, indices = transposed_unraveled)
print(manual_max_pool.shape)
# reshape to be the same as max pool x number of channels
# not yet implemented here for number of channels (later on)
# should be something like
# manual_max_pool = tf.reshape(manual_max_pool, shape=max_pooled.shape.as_list() + [tf_shaped_randint.shape[-1]])
manual_max_pool = tf.reshape(manual_max_pool, shape=max_pooled.shape)
print(np.equal(manual_max_pool, max_pooled))
gather_on_tf_shaped_randint = tf.gather_nd(tf_shaped_randint, transposed_unraveled)
print(gather_on_tf_shaped_randint.shape)
print(gather_on_tf_shaped_randint)
gather_on_tf_shaped_randint = tf.reshape(gather_on_tf_shaped_randint, (4,2,2,2,5))
print(gather_on_tf_shaped_randint.shape)
print(gather_on_tf_shaped_randint)

input('Finish? ')