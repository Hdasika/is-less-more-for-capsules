import tensorflow as tf
import numpy as np
import models
from keras import optimizers, metrics
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

def get_cifar100_dataset(coarse_too=False):
	(X_train, y_train_fine), (X_test, y_test_fine) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
	y_train_fine = to_categorical(y_train_fine, num_classes=100)
	y_test_fine = to_categorical(y_test_fine, num_classes=100)

	y_train_coarse = None
	y_test_coarse = None
	if coarse_too:
		(_, y_train_coarse), (_, y_test_coarse) = tf.keras.datasets.cifar100.load_data(label_mode='coarse')
		y_train_coarse = to_categorical(y_train_coarse, num_classes=20)
		y_test_coarse = to_categorical(y_test_coarse, num_classes=20)

	return [(X_train, X_test), (y_train_fine, y_train_coarse), (y_test_fine, y_test_coarse)] 

def create_data_generator(gen, X, Y_fine, Y_coarse=None, batch_size=8):
	while True:
		# permutation over batch size
		permutations = np.random.permutation(X.shape[0])

		X = X[permutations]
		Y_fine = Y_fine[permutations]
		if Y_coarse:
			Y_coarse = Y_coarse[permutations]

		current_idx = 0
		for X_batched, Y_batched_fine in gen.flow(X, Y_fine, batch_size=batch_size, shuffle=False):
			if Y_coarse:
				until_idx = current_idx + X_batched.shape[0]
				Y_batched_coarse = Y_coarse[current_idx:until_idx]
				yield X_batched, [Y_batched_coarse, Y_batched_fine]
			else:
				yield X_batched, Y_batched_fine
			current_idx += X_batched.shape[0]
			if current_idx >= X.shape[0]:
				break

def prepare_for_trial_model_one(args):
	dataset = get_cifar100_dataset(coarse_too=True)
	datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
	# fit on X
	datagen.fit(dataset[0][0])
	gen = create_data_generator(datagen, dataset[0][0], dataset[1][0],
															Y_coarse=dataset[1][1], batch_size=args.batch_size)
	model = models.TrialModelOne()
	adam = optimizers.Adam()
	model_metrics = {
		'superclass_out': metrics.categorical_accuracy,
		'subclass_out': metrics.categorical_accuracy
	}
	loss = 'categorical_crossentropy'
	loss_weights = [0.2, 0.8]
	model.compile(optimizer=adam, metrics=model_metrics, loss=loss, loss_weights=loss_weights)
	return dataset, gen, model

def prepare_for_trial_model_two(args):
	dataset = get_cifar100_dataset()
	datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
	# fit on X
	datagen.fit(dataset[0][0])
	gen = create_data_generator(datagen, dataset[0][0], dataset[1][0], batch_size=args.batch_size)
	model = models.TrialModelTwo()
	adam = optimizers.Adam()
	metrics = ['accuracy']
	loss = 'categorical_crossentropy'
	model.compile(optimizer=adam, metrics=metrics, loss=loss)
	return dataset, gen, model