from capsule_max_pool import CapsMaxPool
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf

# tf.enable_eager_execution()

shape = (1,4,4,7,8)
x = Input(shape=shape[1:])
maxpool = CapsMaxPool()(x)
model = Model(inputs=x, outputs=maxpool)
input_x = np.random.rand(*shape) * 100
input_y = np.random.rand(*(1,2,2,7,8)) * 100
tensor_x = tf.cast(input_x, dtype=tf.float32)
tensor_y = tf.cast(input_y, dtype=tf.float32)
opt = Adam()
model.compile(optimizer=opt, loss='mean_squared_error')
print(model.fit(x=input_x, y=input_y, batch_size=1))
