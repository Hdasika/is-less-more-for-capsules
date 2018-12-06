import tensorflow as tf

def non_saturating_squash(input_tensor):
	norm = tf.norm(input_tensor, axis=-1, keepdims=True)
	norm_squared = norm * norm
	return (input_tensor / norm) * (norm_squared / (1 + norm))
