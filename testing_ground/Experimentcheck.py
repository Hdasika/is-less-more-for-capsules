import args as args
import keras
import tensorflow as tf
import keras.backend as K
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from keras import layers, callbacks, models, optimizers
from keras.models import Sequential
from SegCaps.capsule_layers import Length, ConvCapsuleLayer, Mask
from SegCaps.custom_losses import margin_loss

channels = 3
kernel_size = 5
height, width , capsules , atoms =(32, 32,  4, 16)
data= tf.keras.datasets.cifar100.load_data(label_mode='fine')
x_train = data[0][0] # 50K  images and each row is one image
y_train = data[0][1] # lables
print(x_train.shape)
x_train1 = tf.cast(x_train, tf.float32)
num_classes=''
#Remember to turn the labels to categorical, Keras has a utility function for that it seems https://keras.io/utils/#to_categorical
y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)


# Havent changed the random.rand( is it correct)?
conv = K.conv2d(x_train1, kernel=tf.constant(np.random.rand(5,5,3,4), dtype=tf.float32), strides=(2,2), padding='same')
tf.constant(np.random.rand(5,5,3,4), dtype=tf.float32)
x = layers.Input(shape=(32,32,3))
conv1 = layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1')(x)
_, H, W, C = conv1.get_shape()
conv1.get_shape()
reshaped_two = tf.reshape(x_train1, [50000, 16, atoms,capsules,channels])
print(reshaped_two.shape)
reshaped_two.set_shape((None, 16, 16, 4,3))

#Layers were put but im not sure what im dealing with properly
#===========================
primary_caps = ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms=16, strides=2, padding='same',
                                    routings=1, name='primarycaps')(reshaped_two)
conv_cap_2_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1, padding='same',
                                    routings=3, name='conv_cap_2_1')(primary_caps)
conv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=2, padding='same',
                                    routings=3, name='conv_cap_2_2')(conv_cap_2_1)
digitcaps = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                    routings=3, name='conv_cap_3_1')(conv_cap_2_2)
#============================
# have taken this from Rodney's
_, H, W, C, A = digitcaps.get_shape()
y = layers.Input(shape=(32,32,3))
masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction
out_caps = Length(num_classes=3,name='capsnet')(digitcaps)
#====================================

n_class=''
input_shape=''
#Decoder
#=============
decoder = Sequential(name='decoder')
decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
decoder.add(layers.Dense(1024, activation='relu'))
decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))
#==============

train_model = models.Model(inputs=[x, y], outputs=[out_caps, decoder(masked_by_y)],)
eval_model = models.Model(inputs=x, outputs=[out_caps, decoder(masked)])

def train_generator(x, y, batch_size, shift_fraction=0.):
    train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                       height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
    generator = train_datagen.flow(x, y, batch_size=batch_size)
    while 1:
        x_batch, y_batch = generator.next()
        yield ([x_batch, y_batch], [y_batch, x_batch])
# TENSORBOARD Callback
# the data input format put ,again...im not sure
#=====================
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
train_model.compile(optimizer=optimizers.Adam(lr=args.lr),loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon])
train_model.fit([y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data= [""], callbacks=[tbCallBack])


#Want to normalize input and seems ImageDataGenerator can help us with that https://keras.io/preprocessing/image/#imagedatagenerator-class. Also check out
# https://stackoverflow.com/questions/41855512/how-does-data-normalization-work-in-keras-during-prediction for ideas on how to use it
train_model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[""],
                        callbacks=[tbCallBack])


#this is experimental which is not require for capset.. but thought might help in future
#============================================
# model.add(Conv2D(32, (3, 3), padding='same', activation='relu',
#                  input_shape=x_train.shape[1:]))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
#
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
#               metrics=['accuracy'])
#=====================================================