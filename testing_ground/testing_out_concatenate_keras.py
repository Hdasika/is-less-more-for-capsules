from keras.layers import Dense, Input, Reshape, Concatenate, Flatten
from keras.models import Model
import numpy as np

input_one = Input(shape=(10,5), name='input')
reshaper = Reshape((5,10), name='reshaper')(input_one)
dense_relu = Dense(32, activation='relu', name='dense_relu')(reshaper)
dense_sigmoid = Dense(32, activation='sigmoid', name='dense_sigmoid')(reshaper)
# basically making like ?, 10, 32 here alternatively if axis 2 then ? 5, 64
concatenate = Concatenate(axis=1, name='concatenate')([dense_relu, dense_sigmoid])
flatten = Flatten(data_format='channels_last', name='flatten')(concatenate)
predictions = Dense(1, activation='sigmoid', use_bias=True, name='predictions')(flatten)
model = Model(inputs=input_one, outputs=predictions)
model.compile('sgd', loss='binary_crossentropy', metrics=['accuracy'])
data = np.random.randint(10, size=(1000, 10, 5))
labels = np.random.randint(2, size=(1000,))
model.fit(data, labels, epochs=20)