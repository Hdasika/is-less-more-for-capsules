import tensorflow as tf
import numpy as np  

tf.enable_eager_execution()

batch, atoms, height, width, channels = (1,3,4,4,2)
filter_depth, filter_height, filter_width, filter_channels, filter_out_channel = (1, 2, 2, channels, 6)

data = np.random.randint(6, size=(batch, atoms, height, width, channels))
data = tf.cast(data, tf.float32)
print('Data shape', tf.shape(data))
print('Data content', data)
filter = np.random.randint(3, size=(filter_depth, filter_height, filter_width, filter_channels, filter_out_channel))
filter = tf.cast(filter, tf.float32)
print('Filter content', filter)
print('Filter shape', tf.shape(filter))
broadcasted_filter = tf.broadcast_to(filter, [atoms, filter_height, filter_width, filter_channels, filter_out_channel])
print('Broadcasted filter content', broadcasted_filter)
print('Broadcast filter shape', tf.shape(broadcasted_filter))

result = tf.nn.conv3d(data, broadcasted_filter, strides=(1,1,1,1,1), padding='SAME', data_format='NDHWC')
print('Result shape', tf.shape(result))
print(result)