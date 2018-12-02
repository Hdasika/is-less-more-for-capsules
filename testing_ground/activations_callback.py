## https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer 
## https://github.com/knathanieltucker/a-bit-of-deep-learning-and-keras/blob/master/notebooks/Callbacks.ipynb

## I like the first 2 links, you are free to look at these as well if you want.

## https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer
## https://datascience.stackexchange.com/questions/20469/keras-visualizing-the-output-of-an-intermediate-layer

## Some things that I want to say is that this will simply print and not store as .npy or .npz, one can do that as well using the following piece of code but I wasn't able to figure out the layer name.

"""
layer_name = 'my_layer'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)
"""

from keras import backend as K
from keras.callbacks import LambdaCallback

def activations():
 inp = model.input                                           # input placeholder
 outputs = [layer.output for layer in model.layers]          # all layer outputs
 functor = K.function([inp, K.learning_phase()], outputs )   # evaluation function
 
 # Testing
 test = np.random.random(input_shape)[np.newaxis,...]
 #test = np.random.random(input_shape)[np.newaxis,:]
 layer_outs = functor([test, 1.])
 print (layer_outs)

activations_callback = LambdaCallback(
    on_train_end=activations)

## Feel free to play around and suggest changes. 


## WORK IN PROGRESS


## https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
## https://github.com/knathanieltucker/a-bit-of-deep-learning-and-keras/blob/master/notebooks/Callbacks.ipynb

## I like the first 2 links, you are free to look at these as well if you want.

## https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer
## https://datascience.stackexchange.com/questions/20469/keras-visualizing-the-output-of-an-intermediate-layer

## Some things that I want to say is that this will simply print and not store as .npy or .npz, one can do that as well using the following piece of code but I wasn't able to figure out the layer name.

"""
layer_name = 'my_layer'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)
"""

#from keras import backend as K
#from keras.callbacks import LambdaCallback

#def activations():
 #inp = model.input                                           # input placeholder
 #outputs = [layer.output for layer in model.layers]          # all layer outputs
 #functor = K.function([inp, K.learning_phase()], outputs )   # evaluation function

 # Testing
 #test = np.random.random(input_shape)[np.newaxis,...]
 #test = np.random.random(input_shape)[np.newaxis,:]
 #layer_outs = functor([test, 1.])
 #print (layer_outs)

activations_callback = LambdaCallback(
    on_train_end=activations)

## Feel free to play around and suggest changes.

import numpy as np
from keras.callbacks import LambdaCallback

class RecordActivationsCallback:

   def __init__(self, name):
      self.name = name # name of the layer you want the activation for

   def get_activations(model, model_inputs, layer_name=self.name):
    inp = model.input
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    output = self.name.output

    # we remove the placeholders (Inputs node in Keras). Not the most elegant though..

   if 'input_' not in output.name:
    output = output


   funcs = K.function(inp, output)  # evaluation functions


   list_inputs = [model_inputs, 0.]

   activations = funcs(list_inputs)[0]
   layer_names = output.name

   result = dict(zip(layer_names, activations))

   import numpy as np
   import matplotlib.pyplot as plt
       """
       (1, 26, 26, 32)
       (1, 24, 24, 64)
       (1, 12, 12, 64)
       (1, 12, 12, 64)
       (1, 9216)
       (1, 128)
       (1, 128)
       (1, 10)
       """
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

    return result
   np.save(sys.argv[1], result)


activations_callback = LambdaCallback(
    on_train_end=activations)

'''
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


lass LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

model = Sequential()
model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

history = LossHistory()
model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, callbacks=[history])

print(history.losses)
# outputs
'''
[0.66047596406559383, 0.3547245744908703, ..., 0.25953155204159617, 0.25901699725311789]
'''
-

'''
