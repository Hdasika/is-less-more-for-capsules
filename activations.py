import tensorflow as tf

def non_saturating_squash(input_tensor):
	norm = tf.norm(input_tensor, axis=-1, keepdims=True)
	return input_tensor * (norm / (1 + norm))
