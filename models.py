from keras import layers, models
from layers.coupled_capsule import CoupledConvCapsule
from layers.capsule_max_pool import CapsMaxPool
from layers.capsule_norm import CapsuleNorm
from SegCaps.capsule_layers import ConvCapsuleLayer
import tensorflow as tf

def FullConvolutionalModel():
	pass

def TrialModel():
	################## normal convolution ######################
	input = layers.Input((32,32,3))
	convolutional = layers.Conv2D(filters=8, kernel_size=2, strides=1, padding='same', activation='relu', data_format='channels_last', name='conv')(input)
	############################################################

	################## coupled+conv layers #####################
	_, H, W, C = convolutional.get_shape()
	reshaped_conv = layers.Reshape((H.value, W.value, 1, C.value), name='reshape_conv')(convolutional)
	primary_caps = CoupledConvCapsule(num_capsule_types=4, num_caps_instantiations=12, filter_size=(2,2),
											strides=(1,1), padding='same', routing=2, name='primary_caps')(reshaped_conv)
	caps_conv_1_1 = ConvCapsuleLayer(kernel_size=2, num_capsule=8, num_atoms=12, strides=1,
									 padding='valid', routings=3, name='caps_conv_1_1')(primary_caps)
	coupled_conv_1_1 = CoupledConvCapsule(num_capsule_types=6, num_caps_instantiations=24, filter_size=(4,4),
											strides=(1,1), padding='same', routing=2, name='coupled_conv_1_2')(caps_conv_1_1)
	caps_conv_1_2 = ConvCapsuleLayer(kernel_size=4, num_capsule=10, num_atoms=24, strides=1,
									 padding='valid', routings=3, name='caps_conv_1_2')(coupled_conv_1_1)
	############################################################

	################## coupled+conv layers #####################
	coupled_conv_2_1 = CoupledConvCapsule(num_capsule_types=16, num_caps_instantiations=32, filter_size=(5,5),
											strides=(1,1), padding='same', routing=2, name='coupled_conv_2_1')(caps_conv_1_2)
	caps_conv_2_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=20, num_atoms=32, strides=2,
									 padding='valid', routings=3, name='caps_conv_2_1')(coupled_conv_2_1)
	############################################################

	################## norm output for superclass ###############
	superclass_norm = CapsuleNorm(name='superclass_norm')(caps_conv_2_1)
	superclass_avg_pool = layers.GlobalAveragePooling2D(data_format='channels_last', name='superclass_avg_pool')(superclass_norm)
	superclass_out = layers.Activation('softmax', name='superclass_out')(superclass_avg_pool)
	#############################################################

	##################### End layers ###########################
	conv_last_layers_1 = layers.Conv2D(filters=32, kernel_size=2, padding='valid', activation='relu',
											data_format='channels_last', name='conv_last_1')(superclass_norm)
	conv_last_layers_2 = layers.Conv2D(filters=64, kernel_size=4, padding='valid', activation='relu',
											kernel_initializer='he_normal', data_format='channels_last', name='conv_last_2')(conv_last_layers_1)
	conv_last_layers_3 = layers.Conv2D(filters=128, kernel_size=4, padding='valid', activation='relu',
											kernel_initializer='he_normal', data_format='channels_last', name='conv_last_3')(conv_last_layers_2)
	last_avg_pool = layers.GlobalAveragePooling2D(data_format='channels_last', name='last_avg_pool')(conv_last_layers_3)
	subclass_out = layers.Dense(100, activation='softmax', name='subclass_out')(last_avg_pool)
	#############################################################

	model = models.Model(inputs=input, outputs=[superclass_out, subclass_out])
	return model