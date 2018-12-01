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