import tensorflow as tf
import keras.backend as K
from keras import initializers
from keras.layers import Layer
from keras.utils.conv_utils import conv_output_length
from SegCaps.capsule_layers import update_routing

class CoupledConvCapsule(Layer):
	def __init__(
		self,
		num_capsule_types,
		num_caps_instantiations,
		filter_size=(2,2),
		strides=(2,2),
		padding='same',
		filter_initializer='he_normal',
		routings=3,
		**kwargs
	):
		"""Coupled convolutional capsule
		
		This layer computes a coupled version of convolutional capsule
		A coupled version of convolutional capsule shares the parameters of the kernel
		along the instantiation parameters of a capsule. The idea is that using the same
		weight across the instantiation parameter will enable the network to better learn
		the relationship of the instantiation parameters within a capsule.
		
		Previously in the original capsule networks the relationship of the instantiation parameters
		(or hence neurons) were being captured primarily through the squashing function.
		However there was no explicit learning on the weights of each instantiation parameters
		and how they may be connected to each other.

		Inspired by https://github.com/lalonderodney/SegCaps.
		
		:param num_capsule_types: Number of capsule types to be created in layer
		:type num_capsule_types: int
		:param num_caps_instantiations: Number of instantiation parameters for capsules in this layer
		:type num_caps_instantiations: int
		:param filter_size: Filter size, defaults to (2,2)
		:type filter_size: tuple[int], optional
		:param strides: Convolutional stride, defaults to (2,2)
		:type strides: tuple, optional
		:param padding: Convolutional padding strategy, defaults to 'same'
		:type padding: str, optional
		:param filter_initializer: Filter initializer, defaults to 'he_normal'
		:type filter_initializer: str, optional
		:param routings: Number of routings iterations, defaults to 3
		:type routings: int, optional
		:return: self
		:rtype: CoupledConvCapsule
		"""
		self.filter_size = filter_size
		self.strides = strides
		self.num_capsule_types = num_capsule_types
		self.num_caps_instantiations = num_caps_instantiations
		self.filter_initializer = filter_initializer
		self.padding = padding
		self.routings = routings
		super(CoupledConvCapsule, self).__init__(**kwargs)
	
	def build(self, input_shape):
		assert len(input_shape) == 5, "Input shape is incorrect, should be "\
																	"[Batch size x height x width x capsules x instantiation params]"
		self.input_height = input_shape[1]
		self.input_width = input_shape[2]
		self.input_num_capsule_types = input_shape[3]
		self.input_num_caps_instantiations = input_shape[4]

		weight_shape = [self.filter_size[0], self.filter_size[1], 1, self.num_capsule_types * self.num_caps_instantiations]
		self.weights_per_capsule_type = tf.TensorArray(
			dtype=tf.float32, size=self.input_num_capsule_types, element_shape=tf.TensorShape(weight_shape)
		)
		for i in range(self.input_num_capsule_types):
			weights = self.add_weight(
				name='coupled_conv_caps_kernel_{i}'.format(i=i),
				shape=weight_shape, initializer=self.filter_initializer
			)
			self.weights_per_capsule_type = self.weights_per_capsule_type.write(i, weights)

		self.bias = self.add_weight(
			name='coupled_conv_caps_bias',
			shape=[1, 1, self.num_capsule_types, self.num_caps_instantiations],
			initializer=initializers.constant(0),
		)
		super().build(input_shape)

	def _convolution_body(self, i, input_over_capsule_types, convs):
		"""Convolution logic

		Main body to run convolution. It's purpose is to be run on each capsule type
		and running the convolution on the batched input per capsule type

		:param i: An incrementing value (until number of capsule types)
		:type i: tf.constant
		:param input_over_capsule_types: Input over capsule types to convolve
		:type input_over_capsule_types: tf.Tensor
		:param convs: Convolutional result to fill in
		:type convs: tf.TensorArray
		"""
		filter = self.weights_per_capsule_type.read(i)
		# broadcast/copy filter
		broadcasted_filter = K.tile(filter, [1, 1, self.input_num_caps_instantiations, 1])
		input_on_one_capsule_type = input_over_capsule_types[i]
		convoluted_input = K.conv2d(
			input_on_one_capsule_type,
			kernel=broadcasted_filter,
			strides=self.strides,
			padding=self.padding,
			data_format='channels_last'
		)
		convs = convs.write(i, convoluted_input)
		return (i + 1, input_over_capsule_types, convs)

	def call(self, input):
		caps_channel_first_input = K.permute_dimensions(input, (3, 0, 1, 2, 4))
		caps_channel_first_dynamic_shape = K.shape(caps_channel_first_input)

		complete_convolution = tf.TensorArray(
			dtype=tf.float32, size=self.input_num_capsule_types
		)
		i = tf.constant(0)
		_, _, complete_convolution = tf.while_loop(
			cond = lambda i, weights, convs: i < self.input_num_capsule_types,
			body = self._convolution_body,
			loop_vars = [i, caps_channel_first_input, complete_convolution],
			return_same_structure=False
		)

		# get the convoluted input as votes
		# votes refer to the prediction vector received from child (previous) capsule layer
		votes = complete_convolution.stack(name='convoluted_input')
		_, _, conv_height, conv_width, _ = votes.get_shape()

		# set votes dimension
		# [batch size x input num caps types x conv height x conv width x num capsule types x num caps instantiations]
		votes = K.reshape(votes, [caps_channel_first_dynamic_shape[1], self.input_num_capsule_types,
															conv_height.value, conv_width.value,
															self.num_capsule_types, self.num_caps_instantiations])

		# the rest of the logic here is reusing from SegCaps
		# logit shape is [num input capsules x batch size x conv height x conv width x num capsule types]
		logit_shape = K.stack([
			caps_channel_first_dynamic_shape[1], caps_channel_first_dynamic_shape[0],
			conv_height.value, conv_width.value, self.num_capsule_types]
		)
		biases_replicated = K.tile(self.bias, [conv_height, conv_width, 1, 1])

		activations = update_routing(
				votes=votes,
				biases=biases_replicated,
				logit_shape=logit_shape,
				num_dims=6,
				input_dim=self.input_num_capsule_types,
				output_dim=self.num_capsule_types,
				num_routing=self.routings)

		return activations	

	def compute_output_shape(self, input_shape):
		height, width = input_shape[1:3]
		convolved_height = conv_output_length(
			input_length=height,
			filter_size=self.filter_size[0],
			padding=self.padding,
			stride=self.strides[0]
		)
		convolved_width = conv_output_length(
			input_length=width,
			filter_size=self.filter_size[1],
			padding=self.padding,
			stride=self.strides[1]
		)
		return (input_shape[0], convolved_height, convolved_width, self.num_capsule_types, self.num_caps_instantiations)
	
	def get_config(self):
		config = {
			'num_capsule_types': self.num_capsule_types,
			'num_caps_instantiations': self.num_caps_instantiations,
			'filter_size': self.filter_size,
			'strides': self.strides,
			'padding': self.padding,
			'filter_initializer': self.filter_initializer,
			'routings': self.routings,
		}
		base_config = super(CoupledConvCapsule, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))