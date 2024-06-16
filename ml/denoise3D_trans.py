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
from ml.custom_keras_layers import TransformerBlock

def MakeDenoise3DTrans(config_path,
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
    num_trans         = config['HYPERPARAMETERS']['denoise_hyperparams_dict']['num_trans']
    
    assert (config['MODEL_TYPE'] == 'DenoiseTrans'), "'MODEL_TYPE' must be 'DenoiseTrans'. Got: "+str(config['MODEL_TYPE'])

    if (num_trans < 0):
        num_trans = 1
    
    chanDim = -1
    #----------------------------------------------------------------
    ''' Encoder '''
    input_ = Input(shape=x_data_shape[1:])
    x = input_
    
    x = ConvLSTM2D(filters=outer_filters,
                   kernel_size=(5, 5), # orig 4, 2
                   strides=(4, 2),  # orig 4, 2
                   padding="same",
                   activation=LeakyReLU(alpha=0.2),
                   recurrent_activation='tanh',
                   return_sequences=False)(x)
    #x = BatchNormalization(axis=chanDim)(x)
    x = LayerNormalization()(x)
    x = Dropout(rate=0.1)(x)

    x = TransformerBlock(embed_dim=x.shape[-1], # Must always be prev_layer.shape[-1]
                         num_heads=4,
                         ff_dim=outer_filters, # Must always be next_layer.shape[-1]
                         attn_axes=(1, 2, 3))(x)   # If input is (Batch, time, d1,...,dn, embed_dim), must be indices of (d1,...,dn)

    pre_latent_size = x.shape
    #---------------------------------------------------------------
    ''' Bottleneck '''

    if (num_latent_layers > 0):
        x = Flatten()(x)

        for i in range(num_latent_layers):
            x = Dense(latent_size, activation='tanh')(x)
            x = LayerNormalization()(x)
            x = Dropout(rate=0.1)(x)
        
        x = Dense(units=y_data_shape[1] * np.prod(pre_latent_size[1:]), activation=LeakyReLU(alpha=0.2))(x)
        x = LayerNormalization()(x)
        x = Dropout(rate=0.1)(x)

        x = Reshape((y_data_shape[1], pre_latent_size[2], pre_latent_size[3], pre_latent_size[4]))(x)
    
    else:
        try: # Conv3D
            x = Reshape((y_data_shape[1], pre_latent_size[2], pre_latent_size[3], pre_latent_size[4]))(x)
        except IndexError: #ConvLSTM2D
            x = Reshape((y_data_shape[1], pre_latent_size[1], pre_latent_size[2], pre_latent_size[3]))(x)
    #----------------------------------------------------------------
    ''' Decoder '''
    
    x = Convolution3DTranspose(filters=outer_filters, 
                               kernel_size=(y_data_shape[1], 5, 5),
                               strides=(y_data_shape[1], 4, 2),
                               padding="same",
                               activation=LeakyReLU(alpha=0.2))(x)
    #x = BatchNormalization(axis=chanDim)(x)
    x = LayerNormalization()(x)

    output = Conv3D(filters=y_data_shape[4], 
                    kernel_size=(1, 1, 1),
                    strides=(1, 1, 1),
                    padding="same",
                    activation='linear')(x)
    #----------------------------------------------------------------
    model = keras.Model(input_, output, name="Denoise3DTrans")
    
    print(model.summary())
    keras.utils.plot_model(model, show_shapes=True, show_layer_activations=True, to_file=os.path.join('SavedModels/Figs', 'Denoise3D1StageTrans.png'))

    return model
#====================================================================
# x_train = np.random.rand(100, 5, 28, 14, 8)
# y_train = np.random.rand(100, 1, 28, 14, 1)

# autoencoder = MakeDenoise('config.yml',
#                           x_data_shape=x_train.shape,
#                           y_data_shape=y_train.shape)

# autoencoder.compile(loss='mse',optimizer=keras.optimizers.Adam(learning_rate=0.001))

# history = autoencoder.fit(x=x_train,
#                           y=y_train,
#                           epochs=10,
#                           batch_size=32)