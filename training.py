import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras import backend as K
from keras.utils import to_categorical
from keras import layers, models, optimizers
from layers.coupled_capsule import CoupledConvCapsule
import models
import numpy as np
#from TrainModel import TrialModel # import the TrailModel

# 1. Standardize images across the dataset, mean=0, stdev=1
'''standardize pixel values across the entire dataset'''

# K.set_image_dim_ordering('th')
# load data
(X_train, y_train_fine), (X_test, y_test_fine) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
fine_classes = 100

(_, y_train_coarse), (_, y_test_coarse) = tf.keras.datasets.cifar100.load_data(label_mode='coarse')
coarse_classes = 20

y_train_fine = to_categorical(y_train_fine, num_classes=fine_classes)
y_test_fine = to_categorical(y_test_fine, num_classes=fine_classes)

y_train_coarse = to_categorical(y_train_coarse, num_classes=coarse_classes)
y_test_coarse = to_categorical(y_test_coarse, num_classes=coarse_classes)

# define data preparation
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
# fit centering and normalization from data
datagen.fit(X_train)

'''Point of Comparison for Image Augmentation'''
# configure batch size and retrieve one batch of images
# for X_batch, y_batch in datagen.flow(X_train, [y_train, y_train_coarse], batch_size=9):
	#print(X_batch[0], y_batch)
	# # create a grid of 3x3 images
	# for i in range(0, 9):
	# 	pyplot.subplot(330 + 1 + i)
	# 	pyplot.imshow(X_batch[i].astype(np.int32), cmap=pyplot.get_cmap('brg'), interpolation='nearest')
	# # show the plot
	# pyplot.show()
	# break

batch_size = 64
def createDataGenerator(gen, X, Y_fine, Y_coarse):
	global batch_size
	while True:
		# permutation over batch sze
		permutations = np.random.permutation(X.shape[0])

		X = X[permutations]
		Y_fine = Y_fine[permutations]
		Y_coarse = Y_coarse[permutations]

		current_idx = 0
		for X_batched, Y_batched_fine in gen.flow(X, Y_fine, batch_size=batch_size, shuffle=False):
			until_idx = current_idx + X_batched.shape[0]
			Y_batched_coarse = Y_coarse[current_idx:until_idx]
			current_idx += until_idx

			yield X_batched, [Y_batched_coarse, Y_batched_fine]

generator_for_training = createDataGenerator(datagen, X_train, y_train_fine, y_train_coarse)
model = models.TrialModel()
model.summary(line_length=150)

adam = optimizers.Adam()
metrics = {'superclass_out': 'accuracy', 'subclass_out': 'accuracy'}
loss_weights = [0.2, 0.8]
loss = 'categorical_crossentropy'

model.compile(optimizer=adam, metrics=metrics, loss=loss, loss_weights=loss_weights)

try:
	train_history = model.fit_generator(generator_for_training, epochs=50,
		steps_per_epoch=X_train.shape[0] / batch_size, verbose=1, maximum_queue_size=20, workers=2)
except KeyboardInterrupt:
	print('Keyboard interrupted during training...')
finally:
	model.save('saved_trial_model.h5')