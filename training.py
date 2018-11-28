import tensorflow as tf 
import utils
import models
import argparse

parser = argparse.ArgumentParser(description='CMPT726 Project')
parser.add_argument('--model_series', metavar='model', type=int, required=True, choices=[1,2,3], help='Choose model series')
parser.add_argument('--save_dest', metavar='dest', type=str, required=True, help='Save model destination')
parser.add_argument('--batch_size', metavar='bs', type=int, required=False, default=8, help='Batch size')
parser.add_argument('--epochs', metavar='e', type=int, required=False, default=50, help='Epochs')
parser.add_argument('--workers', metavar='w', type=int, required=False, default=1, help='Number of workers')

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

# parse sys.argv
args = parser.parse_args()

if args.model_series == 1:
	dataset, gen, model = utils.prepare_for_model(models.TrialModelOne, args, coarse_too=True)
elif args.model_series == 2:
	dataset, gen, model = utils.prepare_for_model(models.TrialModelTwo, args)
elif args.model_series == 3:
	dataset, gen, model = utils.prepare_for_model(models.TrialModelThree, args, coarse_too=True)

model.summary()

try:
	train_history = model.fit_generator(gen, epochs=args.epochs,
		steps_per_epoch=dataset[0][0].shape[0] / args.batch_size, verbose=1, max_queue_size=10, workers=args.workers)
except KeyboardInterrupt:
	print('Keyboard interrupted during training...')
finally:
	model.save(args.save_dest)