import argparse
from layers.margin_loss import margin_loss

parser = argparse.ArgumentParser(description='CMPT726 Project')
parser.add_argument('--model_series', metavar='model', type=int, required=True, choices=[1,2,3,4,5,6,7,8,9], help='Choose model series')
parser.add_argument('--save_dest', metavar='dest', type=str, required=True, help='Save model destination')
parser.add_argument('--batch_size', metavar='bs', type=int, required=False, default=8, help='Batch size')
parser.add_argument('--epochs', metavar='e', type=int, required=False, default=50, help='Epochs')
parser.add_argument('--lr', metavar='lr', type=float, required=False, default=0.0001, help='Learning rate')
parser.add_argument('--super_loss_weight', metavar='sup_w', type=float, required=False, default=0.2, help='Loss weight for superclass')
parser.add_argument('--sub_loss_weight', metavar='sub_w', type=float, required=False, default=0.8, help='Loss weight for subclass')
parser.add_argument('--gray', required=False, action='store_true', help='Turn images to RGB first or not')
parser.add_argument('--init', required=False, type=str, default='he_normal', choices=['glorot_uniform','he_normal'], help='Kernel initializers')
parser.add_argument('--val_split', type=float, required=False, default=0.1, help='Validation split')
parser.add_argument('-tb', '--tensorboard', required=False, action='store_true', help='Use tensorboard or not')
parser.add_argument('--tb_dir', type=str, required=False, default='./tensorboard', help='Tensorboard directory (only applies if -tb is given)')
parser.add_argument('--tb_rate', type=int, required=False, default=1000, help='Tensorboard update rate')
parser.add_argument('--workers', metavar='w', type=int, required=False, default=1, help='Number of workers')

# parse sys.argv
args = parser.parse_args()

import utils
import models
from keras import callbacks


def get_loss(root, split, net, recon_wei, choice):
    if choice == 'mar':
        loss = margin_loss(margin=0.4, downweight=0.5, pos_weight=1.0)
    else:
        raise Exception("Unknow loss_type")

    if net.find('caps') != -1:
        return {'out_seg': loss, 'out_recon': 'mse'}, {'out_seg': 1., 'out_recon': recon_wei}
    else:
        return loss, None

'''Point of Comparison for Image Augmentation'''
# configure batch size and retrieve one batch of images

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
elif args.model_series == 8:
	dataset, gen, model = utils.prepare_for_model(models.TrialModelEight, args)
elif args.model_series == 9:
	dataset, gen, model = utils.prepare_for_model(models.TrialModelNine, args)


model.summary(line_length=150)

try:
	cbs = []

	if args.tensorboard:
		print(f'Will record for tensorboard to {args.tb_dir}')
		tb = callbacks.TensorBoard(log_dir=args.tb_dir, write_graph=False, write_grads=True,
					histogram_freq=1, batch_size=args.batch_size, update_freq=args.tb_rate)
		cbs.append(tb)
	
	if dataset['y_coarse']['val'] is not None:
		validation_data=(dataset['X']['val'], [dataset['y_coarse']['val'], dataset['y_fine']['val']])
	else:
		validation_data=(dataset['X']['val'], dataset['y_fine']['val'])

	train_history = model.fit_generator(gen, epochs=args.epochs,
		steps_per_epoch=dataset['X']['train'].shape[0] / args.batch_size,
		validation_data=validation_data, verbose=1, max_queue_size=10,
		workers=args.workers, callbacks=cbs)

except KeyboardInterrupt:
	print('Keyboard interrupted during training...')
finally:
	model.save(args.save_dest)