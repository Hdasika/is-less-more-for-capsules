from keras import layers, models
from layers.coupled_capsule import CoupledConvCapsule
from layers.capsule_max_pool import CapsMaxPool
from layers.capsule_norm import CapsuleNorm
import layers.capsule as caps
from SegCaps.capsule_layers import ConvCapsuleLayer
from types import SimpleNamespace

def FullConvolutionalModel():
	pass

def TrialModelOne(args):
	################## normal convolution ######################
	input = layers.Input((32,32,3 if not args.gray else 1))
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

def TrialModelTwo(args):
	################## normal convolution ######################
	input = layers.Input((32,32,3 if not args.gray else 1))
	convolutional = layers.Conv2D(filters=8, kernel_size=2, strides=1, padding='same', activation='relu', data_format='channels_last', name='conv')(input)
	############################################################

	################## conv layers #####################
	_, H, W, C = convolutional.get_shape()
	reshaped_conv = layers.Reshape((H.value, W.value, 1, C.value), name='reshape_conv')(convolutional)
	primary_caps = ConvCapsuleLayer(kernel_size=2, num_capsule=4, num_atoms=8, strides=1,
									 padding='same', routings=1, name='primary_caps')(reshaped_conv)
	caps_conv_1 = ConvCapsuleLayer(kernel_size=2, num_capsule=8, num_atoms=12, strides=1,
									 padding='same', routings=3, name='caps_conv_1')(primary_caps)
	caps_conv_2 = ConvCapsuleLayer(kernel_size=2, num_capsule=16, num_atoms=16, strides=2,
									 padding='same', routings=3, name='caps_conv_2')(caps_conv_1)
	caps_conv_3 = ConvCapsuleLayer(kernel_size=4, num_capsule=32, num_atoms=20, strides=1,
									 padding='same', routings=3, name='caps_conv_3')(caps_conv_2)
	caps_conv_4 = ConvCapsuleLayer(kernel_size=4, num_capsule=48, num_atoms=24, strides=1,
									 padding='same', routings=3, name='caps_conv_4')(caps_conv_3)
	caps_conv_5 = ConvCapsuleLayer(kernel_size=5, num_capsule=64, num_atoms=32, strides=2,
									 padding='same', routings=3, name='caps_conv_5')(caps_conv_4)
	caps_conv_6 = ConvCapsuleLayer(kernel_size=5, num_capsule=100, num_atoms=32, strides=1,
									 padding='same', routings=3, name='caps_conv_6')(caps_conv_5)
	############################################################

	################## norm output for superclass ###############
	subclass_norm = CapsuleNorm(name='subclass_norm')(caps_conv_6)
	subclass_avg_pool = layers.GlobalAveragePooling2D(data_format='channels_last', name='subclass_avg_pool')(subclass_norm)
	subclass_out = layers.Activation('softmax', name='subclass_out')(subclass_avg_pool)
	#############################################################

	model = models.Model(inputs=input, outputs=subclass_out)
	return model

def TrialModelThree(args):
	################## normal convolution ######################
	input = layers.Input((32,32,3 if not args.gray else 1))
	convolutional = layers.Conv2D(filters=8, kernel_size=2, strides=1, padding='same', activation='relu', data_format='channels_last', name='conv')(input)
	############################################################

	################## conv layers #####################
	_, H, W, C = convolutional.get_shape()
	reshaped_conv = layers.Reshape((H.value, W.value, 1, C.value), name='reshape_conv')(convolutional)
	primary_caps = ConvCapsuleLayer(kernel_size=2, num_capsule=4, num_atoms=8, strides=1,
									 padding='same', routings=1, name='primary_caps')(reshaped_conv)
	caps_conv_1 = ConvCapsuleLayer(kernel_size=2, num_capsule=8, num_atoms=12, strides=2,
									 padding='same', routings=3, name='caps_conv_1')(primary_caps)
	caps_conv_2 = ConvCapsuleLayer(kernel_size=2, num_capsule=12, num_atoms=16, strides=1,
									 padding='same', routings=3, name='caps_conv_2')(caps_conv_1)
	caps_conv_3 = ConvCapsuleLayer(kernel_size=3, num_capsule=12, num_atoms=24, strides=1,
									 padding='same', routings=3, name='caps_conv_3')(caps_conv_2)
	caps_conv_4 = ConvCapsuleLayer(kernel_size=3, num_capsule=16, num_atoms=24, strides=2,
									 padding='same', routings=3, name='caps_conv_4')(caps_conv_3)
	caps_conv_5 = ConvCapsuleLayer(kernel_size=5, num_capsule=20, num_atoms=32, strides=1,
									 padding='same', routings=3, name='caps_conv_5')(caps_conv_4)
	############################################################

	################## norm output for superclass ###############
	superclass_norm = CapsuleNorm(name='superclass_norm')(caps_conv_5)
	superclass_avg_pool = layers.GlobalAveragePooling2D(data_format='channels_last', name='superclass_avg_pool')(superclass_norm)
	superclass_out = layers.Activation('softmax', name='superclass_out')(superclass_avg_pool)
	#############################################################

	##################### End layers ###########################
	conv_last_layers_1 = layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu',
											data_format='channels_last', name='conv_last_1')(superclass_norm)
	conv_last_layers_2 = layers.Conv2D(filters=64, kernel_size=5, padding='same', activation='relu',
											kernel_initializer='he_normal', data_format='channels_last', name='conv_last_2')(conv_last_layers_1)
	conv_last_layers_3 = layers.Conv2D(filters=96, kernel_size=5, padding='same', activation='relu',
											kernel_initializer='he_normal', data_format='channels_last', name='conv_last_3')(conv_last_layers_2)
	conv_last_layers_4 = layers.Conv2D(filters=128, kernel_size=7, strides=2, padding='same', activation='relu',
											kernel_initializer='he_normal', data_format='channels_last', name='conv_last_4')(conv_last_layers_3)
	last_avg_pool = layers.GlobalAveragePooling2D(data_format='channels_last', name='last_avg_pool')(conv_last_layers_4)
	subclass_out = layers.Dense(100, activation='softmax', name='subclass_out')(last_avg_pool)
	#############################################################

	model = models.Model(inputs=input, outputs=[superclass_out, subclass_out])
	return model

def TrialModelFour(args):
	################## normal convolution ######################
	input = layers.Input((32,32,3 if not args.gray else 1))
	convolutional = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu', data_format='channels_last', name='conv')(input)
	############################################################

	################## conv layers #####################
	_, H, W, C = convolutional.get_shape()
	reshaped_conv = layers.Reshape((H.value, W.value, 1, C.value), name='reshape_conv')(convolutional)
	primary_caps = ConvCapsuleLayer(kernel_size=3, num_capsule=8, num_atoms=32, strides=1,
									 padding='valid', routings=1, name='primary_caps')(reshaped_conv)
	caps_conv_1 = ConvCapsuleLayer(kernel_size=3, num_capsule=16, num_atoms=48, strides=1,
									 padding='valid', routings=1, name='caps_conv_1')(primary_caps)
	caps_conv_2 = ConvCapsuleLayer(kernel_size=3, num_capsule=16, num_atoms=64, strides=1,
									 padding='valid', routings=1, name='caps_conv_2')(caps_conv_1)
	caps_conv_3 = ConvCapsuleLayer(kernel_size=3, num_capsule=20, num_atoms=96, strides=1,
									 padding='valid', routings=1, name='caps_conv_3')(caps_conv_2)
	############################################################

	################# norm output for superclass ###############
	superclass_norm = CapsuleNorm(name='superclass_norm')(caps_conv_3)
	superclass_out = layers.GlobalAveragePooling2D(data_format='channels_last', name='superclass_out')(superclass_norm)
	############################################################

	##################### End layers ###########################
	caps_conv_merge = ConvCapsuleLayer(kernel_size=3, num_capsule=1, num_atoms=192, strides=2,
									padding='same', routings=3, name='caps_conv_merge')(caps_conv_3)
	_, H, W, C, A = caps_conv_merge.get_shape()
	reshaped_caps_conv_merge = layers.Reshape((H.value, W.value, A.value), name='reshaped_caps_conv_merge')(caps_conv_merge)
	conv_last_layers_1 = layers.Conv2D(filters=192, kernel_size=3, strides=2, padding='same', activation='relu',
											data_format='channels_last', name='conv_last_1')(reshaped_caps_conv_merge)
	conv_last_layers_2 = layers.Conv2D(filters=192, kernel_size=1, padding='same', activation='relu',
											data_format='channels_last', name='conv_last_2')(conv_last_layers_1)
	conv_last_layers_3 = layers.Conv2D(filters=10, kernel_size=1, padding='same', activation='relu',
											data_format='channels_last', name='conv_last_3')(conv_last_layers_2)
	last_avg_pool = layers.GlobalAveragePooling2D(data_format='channels_last', name='last_avg_pool')(conv_last_layers_3)
	subclass_out = layers.Dense(100, activation='softmax', name='subclass_out')(last_avg_pool)
	#############################################################

	model = models.Model(inputs=input, outputs=[superclass_out, subclass_out])
	return model

def TrialModelFive(args):
	# Just basic convolution from https://arxiv.org/pdf/1412.6806.pdf model C

	################## normal convolution ######################
	input = layers.Input((32,32,3 if not args.gray else 1))
	conv_1 = layers.Conv2D(filters=96, kernel_size=3, padding='same', activation='relu', data_format='channels_last', name='conv_1')(input)
	conv_2 = layers.Conv2D(filters=96, kernel_size=3, padding='same', activation='relu', data_format='channels_last', name='conv_2')(conv_1)
	max_pool_1 = layers.MaxPool2D(pool_size=3, strides=2, padding='same', name='max_pool_1')(conv_2)
	conv_3 = layers.Conv2D(filters=192, kernel_size=3, padding='same', activation='relu', data_format='channels_last', name='conv_3')(max_pool_1)
	conv_4 = layers.Conv2D(filters=192, kernel_size=3, padding='same', activation='relu', data_format='channels_last', name='conv_4')(conv_3)
	max_pool_2 = layers.MaxPool2D(pool_size=3, strides=2, padding='same', name='max_pool_2')(conv_4)
	conv_5 = layers.Conv2D(filters=192, kernel_size=3, padding='valid', activation='relu', data_format='channels_last', name='conv_5')(max_pool_2)
	conv_6 = layers.Conv2D(filters=192, kernel_size=1, padding='same', activation='relu', data_format='channels_last', name='conv_6')(conv_5)
	conv_7 = layers.Conv2D(filters=10, kernel_size=1, padding='same', activation='relu', data_format='channels_last', name='conv_7')(conv_6)
	avg_pool = layers.GlobalAveragePooling2D(data_format='channels_last', name='avg_pool')(conv_7)
	subclass_out = layers.Dense(100, activation='softmax', name='subclass_out')(avg_pool)
	#############################################################

	model = models.Model(inputs=input, outputs=subclass_out)
	return model

def TrialModelSix(args):
	# similar to 5 but with convolutional capsules
	# there seems to be a bug with keras and my implementation of capsule max pooling
	# https://github.com/keras-team/keras/issues/11753
	# so we go with all convolutional

	################## convolutional caps ######################
	input = layers.Input((32,32,3 if not args.gray else 1))
	convolutional = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', name='convolution')(input)
	primary_caps = caps.PrimaryCap(convolutional, dim_capsule=8, n_channels=32,
									 kernel_size=3, strides=1, padding='valid', initializer=args.init)
	caps_conv_1_1 = ConvCapsuleLayer(kernel_size=3, num_capsule=32, num_atoms=12, strides=1,
									 padding='same', routings=3, name='caps_conv_1')(primary_caps)
	caps_conv_stride2_1 = CoupledConvCapsule(num_capsule_types=24, num_caps_instantiations=16, filter_size=(3,3),
									 padding='valid', filter_initializer=args.init, routings=3, name='caps_conv_stride2_1')(caps_conv_1_1)
	caps_conv_2_1 = ConvCapsuleLayer(kernel_size=3, num_capsule=32, num_atoms=16, strides=1,
									 padding='same', routings=3, name='caps_conv_2_1')(caps_conv_stride2_1)
	caps_conv_stride2_2 = CoupledConvCapsule(num_capsule_types=24, num_caps_instantiations=24, filter_size=(3,3),
									 padding='valid', filter_initializer=args.init, routings=3, name='caps_conv_stride2_2')(caps_conv_2_1)

	caps_conv_3_1 = ConvCapsuleLayer(kernel_size=1, num_capsule=20, num_atoms=24, strides=1,
									 padding='valid', routings=1, name='caps_conv_3_1')(caps_conv_stride2_2)
	caps_conv_3_2 = ConvCapsuleLayer(kernel_size=1, num_capsule=100, num_atoms=28, strides=1,
									 padding='valid', routings=1, name='caps_conv_3_2')(caps_conv_3_1)
	############################################################

	####################### end layer predictions ###########################
	caps_norm = CapsuleNorm(name='caps_norm')(caps_conv_3_2)
	superclass_out = layers.GlobalAveragePooling2D(data_format='channels_last', name='superclass_out')(caps_norm)
	############################################################

	model = models.Model(inputs=input, outputs=superclass_out)
	return model

def TrialModelSeven(args):
	# Much simpler model, similar to Hinton's origina

	################## convolutional ######################
	input = layers.Input((32,32,3 if not args.gray else 1))
	convolutional = layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='same',
															  activation='relu', data_format='channels_last', name='conv')(input)
	_, H, W, C = convolutional.get_shape()
	reshaped_conv = layers.Reshape((H.value, W.value, 1, C.value), name='reshape_conv')(convolutional)

	################# convolutional caps ##################
	primary_caps = ConvCapsuleLayer(kernel_size=3, num_capsule=32, num_atoms=8, strides=1,
									 padding='valid', routings=1, name='primary_caps')(reshaped_conv)
	caps_conv_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=48, num_atoms=12, strides=2,
									 padding='valid', routings=2, name='caps_conv_1')(primary_caps)
	caps_conv_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=64, num_atoms=18, strides=1,
									 padding='valid', routings=2, name='caps_conv_2')(caps_conv_1)
	caps_conv_3 = ConvCapsuleLayer(kernel_size=3, num_capsule=128, num_atoms=24, strides=2,
									 padding='valid', routings=3, name='caps_conv_3')(caps_conv_2)
	caps_conv_4 = ConvCapsuleLayer(kernel_size=4, num_capsule=100, num_atoms=32, strides=1,
									 padding='valid', routings=3, name='caps_conv_4')(caps_conv_3)
	############################################################

	####################### end layer predictions ###########################
	caps_norm = CapsuleNorm(name='caps_norm')(caps_conv_4)
	subclass_out = layers.Flatten(data_format='channels_last', name='subclass_out')(caps_norm)
	############################################################

	model = models.Model(inputs=input, outputs=subclass_out)
	return model

def TrialModelEight(args):
	# Like model seven but with Full-Connected (FC) capsule at the end and a more
	# appropriate PrimaryCaps layer

	################## convolutional ######################
	input = layers.Input((32,32,3 if not args.gray else 1))
	convolutional = layers.Conv2D(filters=512, kernel_size=7, strides=1, padding='valid',
															  activation='relu', data_format='channels_last', name='conv')(input)

	################# primary caps ########################
	primary_caps = caps.PrimaryCap(convolutional, dim_capsule=8, n_channels=64,
									 kernel_size=7, strides=1, padding='valid', initializer=args.init)
	#######################################################

	# ################# convolutional caps ##################
	# coupled_caps_conv_1 = CoupledConvCapsule(num_capsule_types=32, num_caps_instantiations=32, filter_size=(5,5),
	# 										strides=(1,1), padding='same', routing=2, name='coupled_caps_conv_1')(primary_caps)

	caps_conv_1 = ConvCapsuleLayer(kernel_size=7, num_capsule=32, num_atoms=16, strides=1,
									 padding='valid', routings=3, name='caps_conv_1', kernel_initializer=args.init)(primary_caps)
	caps_conv_2 = ConvCapsuleLayer(kernel_size=9, num_capsule=32, num_atoms=24, strides=1,
									 padding='valid', routings=3, name='caps_conv_2', kernel_initializer=args.init)(caps_conv_1)

	# ############################################################

	# ####################### end layer predictions ##############
	flatten_caps = layers.Reshape(target_shape=(-1, 16), name='flatten_caps')(caps_conv_2)
	subclass_prediction_caps = caps.CapsuleLayer(num_capsule=100, dim_capsule=32, routings=3,
									 kernel_initializer=args.init, name='subclass_prediction_caps')(flatten_caps)
	subclass_out = caps.Length(name='subclass_out')(subclass_prediction_caps)
	# ############################################################

	model = models.Model(inputs=input, outputs=subclass_out)
	return model

def TrialModelNine(args):
	# Trying out similar structure as https://arxiv.org/pdf/1805.11195.pdf

	################## convolutional ######################
	input = layers.Input((32,32,3 if not args.gray else 1))
	convolutional = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid',
															  activation='relu', data_format='channels_last', name='conv')(input)

	################# primary caps ########################
	primary_caps = caps.PrimaryCap(convolutional, dim_capsule=16, n_channels=32,
									 kernel_size=9, strides=2, padding='valid', initializer=args.init, to_flatten=True)
	#######################################################

	# ####################### end layer predictions ###########################
	subclass_prediction_caps = caps.CapsuleLayer(num_capsule=100, dim_capsule=32, routings=3,
									 kernel_initializer=args.init, name='subclass_prediction_caps')(primary_caps)
	subclass_out = caps.Length(name='subclass_out')(subclass_prediction_caps)
	# ############################################################

	model = models.Model(inputs=input, outputs=subclass_out)
	return model

def TrialModelTen(args):
	# similar to 6 but with coupled convolutional capsules
	# put together

	################## convolutional caps ######################
	input = layers.Input((32,32,3 if not args.gray else 1))
	convolutional = layers.Conv2D(filters=256, kernel_size=3, strides=1,
									 padding='same', activation='relu', kernel_initializer=args.init, name='convolution')(input)
	primary_caps = caps.PrimaryCap(convolutional, dim_capsule=8, n_channels=32,
									 kernel_size=3, strides=1, padding='valid', initializer=args.init)
	caps_conv_1_1 = ConvCapsuleLayer(kernel_size=3, num_capsule=32, num_atoms=12, strides=1,
									 padding='same', kernel_initializer=args.init, routings=3, name='caps_conv_1')(primary_caps)
	coupled_caps_conv_1 = CoupledConvCapsule(num_capsule_types=24, num_caps_instantiations=16, filter_size=(3,3),
									 padding='valid', filter_initializer=args.init, routings=3, name='coupled_caps_conv_1')(caps_conv_1_1)
	caps_conv_2_1 = ConvCapsuleLayer(kernel_size=3, num_capsule=32, num_atoms=16, strides=1,
									 padding='same', kernel_initializer=args.init, routings=3, name='caps_conv_2_1')(coupled_caps_conv_1)
	coupled_caps_conv_2 = CoupledConvCapsule(num_capsule_types=24, num_caps_instantiations=24, filter_size=(3,3),
									 padding='valid', filter_initializer=args.init, routings=3, name='coupled_caps_conv_2')(caps_conv_2_1)

	caps_conv_3_1 = ConvCapsuleLayer(kernel_size=1, num_capsule=20, num_atoms=24, strides=1,
									 padding='valid', kernel_initializer=args.init, routings=3, name='caps_conv_3_1')(coupled_caps_conv_2)
	caps_conv_3_2 = ConvCapsuleLayer(kernel_size=1, num_capsule=100, num_atoms=28, strides=1,
									 padding='valid', kernel_initializer=args.init, routings=3, name='caps_conv_3_2')(caps_conv_3_1)
	############################################################

	####################### end layer predictions ###########################
	caps_norm = CapsuleNorm(name='caps_norm')(caps_conv_3_2)
	superclass_out = layers.GlobalAveragePooling2D(data_format='channels_last', name='superclass_out')(caps_norm)
	############################################################

	model = models.Model(inputs=input, outputs=superclass_out)
	return model

if __name__ == "__main__":
	model = TrialModelTen(SimpleNamespace(gray=False, init='glorot_uniform'))
	model.summary()