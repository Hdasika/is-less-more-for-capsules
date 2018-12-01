import tensorflow as tf
import numpy as np
from keras import optimizers, metrics
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def convert_rgb_to_gray(images):
  return (0.2125 * images[:,:,:,:1]) + (0.7154 * images[:,:,:,1:2]) + (0.0721 * images[:,:,:,-1:])

def get_cifar100_dataset(args, coarse_too=False):
	(X_train, y_train_fine), (X_test, y_test_fine) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
	if args.gray:
		# make grayscale
		X_train = convert_rgb_to_gray(X_train)

	y_train_coarse = None
	y_val_coarse = None
	y_test_coarse = None
	if coarse_too:
		(_, y_train_coarse), (_, y_test_coarse) = tf.keras.datasets.cifar100.load_data(label_mode='coarse')
		X_train, X_val, y_train_fine, y_val_fine, y_train_coarse, y_val_coarse = train_test_split(
			X_train, y_train_fine, y_train_coarse, train_size=1 - args.val_split, test_size=args.val_split, stratify=y_train_fine
		)
		y_train_coarse = to_categorical(y_train_coarse, num_classes=20)
		y_val_coarse = to_categorical(y_val_coarse, num_classes=20)
		y_test_coarse = to_categorical(y_test_coarse, num_classes=20)
	else:
		X_train, X_val, y_train_fine, y_val_fine = train_test_split(
			X_train, y_train_fine, train_size=1 - args.val_split, test_size=args.val_split, stratify=y_train_fine
		)
	
	y_train_fine = to_categorical(y_train_fine, num_classes=100)
	y_val_fine = to_categorical(y_val_fine, num_classes=100)
	y_test_fine = to_categorical(y_test_fine, num_classes=100)

	return {
		'X': {
			'train': X_train,
			'val': X_val,
			'test': X_test
		},
		'y_fine': {
			'train': y_train_fine,
			'val': y_val_fine,
			'test': y_test_fine
		},
		'y_coarse': {
			'train': y_train_coarse,
			'val': y_val_coarse,
			'test': y_test_coarse
		}
	} 

def create_data_generator(gen, X, Y_fine, Y_coarse=None, batch_size=8):
	while True:
		# permutation over batch size
		permutations = np.random.permutation(X.shape[0])

		X = X[permutations]
		Y_fine = Y_fine[permutations]
		if Y_coarse is not None:
			Y_coarse = Y_coarse[permutations]

		current_idx = 0
		for X_batched, Y_batched_fine in gen.flow(X, Y_fine, batch_size=batch_size, shuffle=False):
			if Y_coarse is not None:
				until_idx = current_idx + X_batched.shape[0]
				Y_batched_coarse = Y_coarse[current_idx:until_idx]
				yield X_batched, [Y_batched_coarse, Y_batched_fine]
			else:
				yield X_batched, Y_batched_fine
			current_idx += X_batched.shape[0]
			if current_idx >= X.shape[0]:
				break

def prepare_for_model(model_fn, args, coarse_too=False):
	dataset = get_cifar100_dataset(args, coarse_too)
	datagen = ImageDataGenerator(rescale=1./255)
	# fit on X
	# datagen.fit(dataset[0][0])
	gen = create_data_generator(datagen, dataset['X']['train'], dataset['y_fine']['train'],
								Y_coarse=dataset['y_coarse']['train'], batch_size=args.batch_size)
	model = model_fn(args)
	opt = optimizers.SGD(lr=args.lr, decay=1e-6)
	loss = 'categorical_crossentropy'

	if coarse_too:
		loss_weights = [args.super_loss_weight, args.sub_loss_weight]
		model.compile(optimizer=opt, metrics=[metrics.categorical_accuracy], loss=loss, loss_weights=loss_weights)
	else:
		model.compile(optimizer=opt, metrics=[metrics.categorical_accuracy], loss=loss)
	return dataset, gen, model