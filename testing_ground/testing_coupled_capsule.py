from layers.coupled_capsule import CoupledConvCapsule
from tensorflow.keras.layers import Input
import tensorflow as tf

input = Input(shape=(12,12,3,8))
coupled_conv_caps = CoupledConvCapsule(
  num_capsule_types=2,
  num_caps_instantiations=12,
  input_caps_instantiations=8,
  filter_size=(2,2),
  strides=(1,1),
)(input)