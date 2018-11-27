from keras import layers
from layers.coupled_capsule import CoupledConvCapsule
from layers.capsule_max_pool import CapsMaxPool
from layers.norm import Norm
from SegCaps.capsule_layers import ConvCapsuleLayer

def FullConvolutionalModel():
	pass

def TrialModel():
	################## normal convolution ######################
	input = layers.Input((32,32,3))
	convolutional = layers.Conv2D(filters=8, kernel_size=2, strides=1, padding='same', activation='relu', data_format='channels_last', name='conv')(input)
	############################################################

	################## coupled+conv layers #####################
	_, H, W, C = convolutional.get_Shape()
	reshaped_conv = layers.Reshape((H.value, W.Value, 1, C.value), name='reshape_conv')(convolutional)
	coupled_conv_1_1 = CoupledConvCapsule(num_capsule_types=2, num_caps_instantiations=12, filter_size=(2,2),
										  strides=(1,1), padding='valid', routing=2, name='coupled_conv_1_1')(reshaped_conv)
	caps_conv_1_1 = ConvCapsuleLayer(kernel_size=(2,2), num_capsule=4, num_atoms=12, strides=1,
									 padding='valid', routings=3, name='caps_conv_1_1')(coupled_conv_1_1)
	coupled_conv_1_2 = CoupledConvCapsule(num_capsule_types=6, num_caps_instantiations=16, filter_size=(4,4),
										  strides=(1,1), padding='valid', routing=2, name='coupled_conv_1_2')(caps_conv_1_1)
	caps_conv_1_2 = ConvCapsuleLayer(kernel_size=(4,4), num_capsule=10, num_atoms=16, strides=1,
									 padding='valid', routings=3, name='caps_conv_1_2')(coupled_conv_1_2)
	############################################################

	################## coupled+conv layers #####################
	coupled_conv_2_1 = CoupledConvCapsule(num_capsule_types=16, num_caps_instantiations=24, filter_size=(5,5),
										  strides=(1,1), padding='valid', routing=2, name='coupled_conv_2_1')(caps_conv_1_2)
	caps_conv_2_1 = ConvCapsuleLayer(kernel_size=(3,3), num_capsule=20, num_atoms=24, strides=1,
									 padding='valid', routings=3, name='caps_conv_2_1')(coupled_conv_2_1)
	############################################################

	################## capsule normalization #################### 
	capsule_norm = Norm(name='capsule_norm')(caps_max)
	capsule_norm = 

	################## subsampling ##############################
	caps_max = CapsMaxPool(pool_size=(2,2), padding='SAME', name='')(caps_conv_2_1)
	#############################################################

	##################### End layers ###########################
	conv_last_layers_1 = layers.Conv2D(filters=32, kernel_size=2, padding='valid', activation='relu', data_format='channels_last', name='conv_last_1')(capsule_norm)
	conv_last_layers_2 = layers.Conv2D(filters=64, kernel_size=2, padding='valid', activation='relu', data_format='channels_last', name='conv_last_1')(capsule_norm)
	#############################################################

