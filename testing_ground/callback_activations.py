import numpy as np
from keras.callbacks import LambdaCallback
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='CMPT726 Project')

mutually_exclusive_options = parser.add_mutually_exclusive_group(required=True)
mutually_exclusive_options.add_argument('--save_activation_mode', type=int,
	choices=[1,2,3], help='Choose save mode')

# 1 for saving after training
# 2 for saving after batch
# 3 for saving after epoch

parser.add_argument('--save_dest', metavar='dest', type=str, required=True, help='Save model destination')

args = parser.parse_args()
args_dict = vars(args)

class RecordActivationsCallback(object):

   def __init__(self, save_activation_mode, save_dest):
      self.name = name # name of the layer you want the activation for
      self.dest = save_dest
      self.mode = save_activation_mode
   
   def get_activations(model, model_inputs, layer_name=self.name):
    
    inp = model.input
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
      
    output = self.name.output

    # we remove the placeholders (Inputs node in Keras). Not the most elegant though..

    if 'input_' not in output.name:
     output = output


    funcs = K.function(inp, output)  # evaluation functions


    list_inputs = [model_inputs, 0.]

    activations = funcs(list_inputs)[0]
    layer_names = output.name

    result = dict(zip(layer_names, activations))

    layer_names = activations.keys()
    activation_maps = activations.values()
    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'One image at a time to visualize.'
    for i, activation_map in enumerate(activation_maps):
        print('Displaying activation map {}'.format(i))
        shape = activation_map.shape
        if len(shape) == 4:
            activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
        elif len(shape) == 2:
            # try to make it square as much as possible. we can skip some activations.
            activations = activation_map[0]
            num_activations = len(activations)
            if num_activations > 1024:  # too hard to display it on the screen.
                square_param = int(np.floor(np.sqrt(num_activations)))
                activations = activations[0: square_param * square_param]
                activations = np.reshape(activations, (square_param, square_param))
            else:
                activations = np.expand_dims(activations, axis=0)
        else:
            raise Exception('len(shape) = 3 has not been implemented.')
        plt.title(layer_names[i])
        plt.imshow(activations, interpolation='None', cmap='jet')
        plt.show()

    #return result
    np.save(self.dest, result)

    if mode is not None:
      if self.mode == 1:
        activations_callback = LambdaCallback(
           on_train_end=get_activations)
      elif self.mode == 2:
        activations_callback = LambdaCallback(
           on_batch_end=get_activations)
      elif self.mode == 3:
        activations_callback = LambdaCallback(
           on_epoch_end=get_activations)
 