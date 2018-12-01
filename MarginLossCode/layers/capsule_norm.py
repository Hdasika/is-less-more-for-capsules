import tensorflow as tf
from keras import layers

class CapsuleNorm(layers.Layer):
	"""Capsule norm layer

	Simply computes the capsules' euclidean norm (or length) and that is the output of this layer
	"""
	def call(self, inputs, **kwargs): #pylint: disable=unused-argument
		assert inputs.get_shape().ndims == 5, 'Input must be a capsule with dimension 5'
		return tf.norm(inputs, axis=-1)

	def compute_output_shape(self, input_shape):
		return input_shape[:-1]

	def get_config(self):
		base_config = super(CapsuleNorm, self).get_config()
		return dict(list(base_config.items()))