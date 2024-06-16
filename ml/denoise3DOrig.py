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


def MakeDenoise(config_path,
                x_data_shape=(1164, 5, 28, 14, 8), 
                y_data_shape=(1164, 1, 28, 14, 1)):
    #----------------------------------------------------------------
    ''' Setup '''
    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    outer_filters     = config['HYPERPARAMETERS']['denoise_hyperparams_dict']['outer_filters']
    middle_filters    = config['HYPERPARAMETERS']['denoise_hyperparams_dict']['middle_filters']
    inner_filters     = config['HYPERPARAMETERS']['denoise_hyperparams_dict']['inner_filters']
    latent_size       = config['HYPERPARAMETERS']['denoise_hyperparams_dict']['latent_size']
    num_latent_layers = config['HYPERPARAMETERS']['denoise_hyperparams_dict']['num_latent_layers']

    assert (config['MODEL_TYPE'] == 'Denoise'), "'MODEL_TYPE' must be 'Denoise'. Got: "+str(config['MODEL_TYPE'])

    chanDim = -1
    #----------------------------------------------------------------
    ''' Encoder '''
    input_ = Input(shape=x_data_shape[1:])
    x = input_
    
    x = Conv3D(filters=outer_filters,
                kernel_size=(3, 3, 3),
                strides=(2, 2, 2),
                padding="same",
                activation=LeakyReLU(alpha=0.2))(x)
    #x = BatchNormalization(axis=chanDim)(x)
    x = LayerNormalization()(x)
    x = Dropout(rate=0.1)(x)

    x = Conv3D(filters=inner_filters,
               kernel_size=(3, 3, 3),
               strides=(x.shape[1], 2, 2),
               padding="same",
               activation=LeakyReLU(alpha=0.2))(x)
    #x = BatchNormalization(axis=chanDim)(x)
    x = LayerNormalization()(x)
    x = Dropout(rate=0.1)(x)
    pre_latent_size = x.shape

    #---------------------------------------------------------------
    ''' Bottleneck '''

    if (num_latent_layers > 0):
        x = Flatten()(x)

        for i in range(num_latent_layers):
            x = Dense(latent_size, activation='tanh')(x)
            x = LayerNormalization()(x)
            x = Dropout(rate=0.1)(x)
        
        x = Dense(units=np.prod(pre_latent_size[1:]), activation=LeakyReLU(alpha=0.2))(x)
        x = LayerNormalization()(x)
        x = Dropout(rate=0.1)(x)

        x = Reshape((pre_latent_size[1], pre_latent_size[2], pre_latent_size[3], pre_latent_size[4]))(x)
    #----------------------------------------------------------------
    ''' Decoder '''
    
    # loop over our number of filters again, but this time in
    # reverse order
    
    x = Convolution3DTranspose(filters=inner_filters,
                               kernel_size=(1, 3, 3),
                               strides=(1, 2, 2),
                               padding="same",
                               output_padding=(0, 1, 0),
                               activation=LeakyReLU(alpha=0.2))(x)
    #x = BatchNormalization(axis=chanDim)(x)
    x = LayerNormalization()(x)
    x = Dropout(rate=0.1)(x)
    # apply a single CONV_TRANSPOSE layer used to recover the
    # original depth of the image
    x = Convolution3DTranspose(filters=outer_filters, 
                               kernel_size=(1, 3, 3),
                               strides=(1, 2, 2),
                               padding="same",
                               activation=LeakyReLU(alpha=0.2))(x)
    #x = BatchNormalization(axis=chanDim)(x)
    x = LayerNormalization()(x)

    output = Convolution3DTranspose(filters=y_data_shape[4], 
                                    kernel_size=(1, 3, 3),
                                    strides=(1, 1, 1),
                                    padding="same",
                                    activation='linear')(x)
    #----------------------------------------------------------------
    
    autoencoder = keras.Model(input_, output, name="autoencoder")
    
    print(autoencoder.summary())
    keras.utils.plot_model(autoencoder, show_shapes=True, show_layer_activations=True, to_file=os.path.join('SavedModels/Figs', 'Denoise3DOrig.png'))

    return autoencoder
#====================================================================
# x_train = np.random.rand(100, 5, 28, 14, 8)
# y_train = np.random.rand(100, 1, 28, 14, 1)

# autoencoder = MakeDenoise('config.yml',
#                           x_data_shape=x_train.shape,
#                           y_data_shape=y_train.shape)
# keras.utils.plot_model(autoencoder, show_shapes=True, show_layer_activations=True, expand_nested=True, to_file=os.path.join('SavedModels/Figs', 'denoise3D.png'))
# print(autoencoder.summary())

# autoencoder.compile(loss='mse',optimizer=keras.optimizers.Adam(lr=0.001))

# history = autoencoder.fit(x=x_train,
#                           y=y_train,
#                           epochs=10,
#                           batch_size=32)