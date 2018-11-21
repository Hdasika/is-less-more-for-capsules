from capsule_max_pool import CapsMaxPool
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.train import AdamOptimizer
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

shape = (1,2,2,2,3)
x = Input(shape=shape[1:])
maxpool = CapsMaxPool()(x)
model = Model(inputs=x, outputs=maxpool)
input_x = np.array([
	[
		[
			[
				[1,2,3],
				[1,2,3]
			],
			[
				[4,5,6],
				[4,5,6]
			]
		],
		[
			[
				[7,8,9],
				[7,8,9]
			],
			[
				[10,11,12],
				[10,11,12]
			]
		],
	]
], dtype=np.float32)
input_y = np.array([
	[
		[
			[
				[10,11,12],
				[10,11,12]
			],
		]
	]
], dtype=np.float32)
tensor_x = tf.cast(input_x, dtype=tf.float32)
tensor_y = tf.cast(input_y, dtype=tf.float32)
opt = AdamOptimizer()
model.compile(optimizer=opt, loss='mean_squared_error')
print(model.fit(x=input_x, y=input_y, batch_size=1))

model.compile(optimizer=opt, loss='mean_squared_error')
