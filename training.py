import argparse

parser = argparse.ArgumentParser(description='CMPT726 Project')
parser.add_argument('--model_series', metavar='model', type=int, required=True, choices=[1,2,3,4,5,6,7], help='Choose model series')
parser.add_argument('--save_dest', metavar='dest', type=str, required=True, help='Save model destination')
parser.add_argument('--batch_size', metavar='bs', type=int, required=False, default=8, help='Batch size')
parser.add_argument('--epochs', metavar='e', type=int, required=False, default=50, help='Epochs')
parser.add_argument('--lr', metavar='lr', type=float, required=False, default=0.0001, help='Learning rate')
parser.add_argument('--super_loss_weight', metavar='sup_w', type=float, required=False, default=0.2, help='Loss weight for superclass')
parser.add_argument('--sub_loss_weight', metavar='sub_w', type=float, required=False, default=0.8, help='Loss weight for subclass')
parser.add_argument('-tb', '--tensorboard', required=False, action='store_true', help='Use tensorboard or not')
parser.add_argument('--tb_dir', type=str, required=False, default='./tensorboard', help='Tensorboard directory (only applies if -tb is given)')
parser.add_argument('--workers', metavar='w', type=int, required=False, default=1, help='Number of workers')

# parse sys.argv
args = parser.parse_args()

import utils
import models
from keras import callbacks

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

if args.model_series == 1:
	dataset, gen, model = utils.prepare_for_model(models.TrialModelOne, args, coarse_too=True)
elif args.model_series == 2:
	dataset, gen, model = utils.prepare_for_model(models.TrialModelTwo, args)
elif args.model_series == 3:
	dataset, gen, model = utils.prepare_for_model(models.TrialModelThree, args, coarse_too=True)
elif args.model_series == 4:
	dataset, gen, model = utils.prepare_for_model(models.TrialModelFour, args, coarse_too=True)
elif args.model_series == 5:
	dataset, gen, model = utils.prepare_for_model(models.TrialModelFive, args)
elif args.model_series == 6:
	dataset, gen, model = utils.prepare_for_model(models.TrialModelSix, args)
elif args.model_series == 7:
	dataset, gen, model = utils.prepare_for_model(models.TrialModelSeven, args)

model.summary(line_length=150)

try:
	cbs = []

	if args.tensorboard:
		print(f'Will record for tensorboard to {args.tb_dir}')
		tb = callbacks.TensorBoard(log_dir=args.tb_dir, write_graph=False,
					batch_size=args.batch_size, update_freq=1000)
		cbs.append(tb)
	
	if cbs:
		train_history = model.fit_generator(gen, epochs=args.epochs,
			steps_per_epoch=dataset[0][0].shape[0] / args.batch_size, verbose=1, max_queue_size=10,
			workers=args.workers, callbacks=cbs)
	else:
		# for some reason google colab can't have tensorboard running...
		train_history = model.fit_generator(gen, epochs=args.epochs,
			steps_per_epoch=dataset[0][0].shape[0] / args.batch_size, verbose=1, max_queue_size=10,
			workers=args.workers)

except KeyboardInterrupt:
	print('Keyboard interrupted during training...')
finally:
	model.save(args.save_dest)