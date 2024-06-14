import sys
sys.dont_write_bytecode = True
import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import netCDF4 as nc
import yaml
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from keras.layers import Input, Concatenate, TimeDistributed, Dense, Conv2D, Conv3D, ConvLSTM2D, LayerNormalization, BatchNormalization, Dropout, AveragePooling3D, UpSampling3D, Reshape, Flatten, LeakyReLU, Convolution2DTranspose, Convolution3DTranspose, Lambda

sys.path.append(os.getcwd())
from ml.ml_utils import Funnel
from ml.custom_keras_layers import RecombineLayer


def MakeDenoise(width, height, depth, filters=(32, 64), latentDim=16):
    inputShape = (height, width, depth)
    chanDim = -1
    # define the input to the encoder
    inputs = Input(shape=inputShape)
    x = inputs
    for f in filters:
        # apply a CONV => RELU => BN operation
        x = Conv2D(filters=f,
                    kernel_size=(3, 3),
                    strides=2,
                    padding="same",
                    activation=LeakyReLU(alpha=0.2))(x)
        # x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(axis=chanDim)(x)
    # flatten the network and then construct our latent vector
    volumeSize = keras.backend.int_shape(x)
    x = Flatten()(x)
    latent = Dense(latentDim)(x)
    # build the encoder model
    encoder = keras.Model(inputs, latent, name="encoder")
    
    latentInputs = Input(shape=(latentDim,))
    x = Dense(np.prod(volumeSize[1:]))(latentInputs)
    x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)
    # loop over our number of filters again, but this time in
    # reverse order
    for f in filters[::-1]:
        # apply a CONV_TRANSPOSE => RELU => BN operation
        x = Convolution2DTranspose(f,
                                   kernel_size=(3, 3),
                                   strides=2,
                                   padding="same",
                                   activation=LeakyReLU(alpha=0.2))(x)
        x = BatchNormalization(axis=chanDim)(x)
    # apply a single CONV_TRANSPOSE layer used to recover the
    # original depth of the image
    outputs = Convolution2DTranspose(filters=depth, 
                                     kernel_size=(3, 3), 
                                     padding="same",
                                     activation='sigmoid')(x)
    # build the decoder model
    decoder = keras.Model(latentInputs, outputs, name="decoder")
    # our autoencoder is the encoder + decoder
    autoencoder = keras.Model(inputs, decoder(encoder(inputs)), name="autoencoder")
    # return a 3-tuple of the encoder, decoder, and autoencoder
    return (encoder, decoder, autoencoder)
#====================================================================
x_train = np.random.rand(100, 12, 12, 3)
y_train = np.random.rand(100, 12, 12, 3)

(encoder, decoder, autoencoder) = MakeDenoise(x_train.shape[1], x_train.shape[2], x_train.shape[3])
keras.utils.plot_model(autoencoder, show_shapes=True, show_layer_activations=True, expand_nested=True, to_file=os.path.join('SavedModels/Figs', 'denoise.png'))
print(autoencoder.summary())

autoencoder.compile(loss='mse',optimizer=keras.optimizers.Adam(lr=0.001))

history = autoencoder.fit(x=x_train,
                          y=y_train,
                          epochs=10,
                          batch_size=32)