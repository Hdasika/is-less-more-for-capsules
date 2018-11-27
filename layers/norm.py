from tensorflow.keras import layers

class Norm(layers.Layer):
	def __init__(self, **kwargs):
		"""Length layer

		Simply computes the capsules' euclidean norm (or length) and that is the output of this layer
		"""
		super(Length, self).__init__(**kwargs)

	def call(self, inputs, **kwargs):
		assert inputs.get_shape().ndims == 5, 'Input must be a capsule with dimension 5'
		return tf.norm(inputs, axis=-1)

	def compute_output_shape(self, input_shape):
		return input_shape[:-2]

	def get_config(self):
		base_config = super(Length, self).get_config()
		return dict(list(base_config.items())