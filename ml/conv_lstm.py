import sys
sys.dont_write_bytecode = True
import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import yaml

import keras_tuner as kt
from tensorflow import keras
from keras.layers import Input, ConvLSTM2D, LayerNormalization, Dropout, Conv3DTranspose, LeakyReLU, Reshape

sys.path.append(os.getcwd())
from ml.custom_keras_layers import TransformerBlock
#====================================================================
def MakeConvLSTM(config_path, 
                 x_data_shape=(1164, 5, 28, 14, 8), 
                 y_data_shape=(1164, 1, 28, 14, 1), 
                 to_file=None):
    #----------------------------------------------------------------
    ''' Setup '''
    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    num_filters       = config['HYPERPARAMETERS']['convlstm_hyperparams_dict']['num_filters']
    num_trans         = config['HYPERPARAMETERS']['convlstm_hyperparams_dict']['num_trans']
    
    region            = config['REGION']

    assert (config['MODEL_TYPE'] == 'ConvLSTM'), "'MODEL_TYPE' must be 'ConvLSTM'. Got: "+str(config['MODEL_TYPE'])

    if (num_trans < 0):
        num_trans = 0
    #----------------------------------------------------------------
    x_strides = 69
    y_strides = 69
    if not (region == 'Whole_Area'):
        x_strides = (4, 2)
        y_strides = (y_data_shape[1], 4, 2)
    else:
        x_strides = (4, 4)
        y_strides = (y_data_shape[1], 4, 4)
    #----------------------------------------------------------------
    ''' Encoder '''
    input_ = Input(shape=x_data_shape[1:])
    x = input_
    
    x = ConvLSTM2D(filters=num_filters, # On website 32
                   kernel_size=(5, 5),
                   strides=x_strides,
                   padding="same",
                   activation=LeakyReLU(alpha=0.2),
                   recurrent_activation='tanh',
                   return_sequences=False)(x)
    
    if (num_trans == 0):
        #x = BatchNormalization(axis=chanDim)(x)
        x = LayerNormalization()(x)
        x = Dropout(rate=0.1)(x)
    else:
        for _ in range(num_trans):
            x = TransformerBlock(embed_dim=x.shape[-1], # Must always be prev_layer.shape[-1]
                                 num_heads=4,
                                 ff_dim=x.shape[-1], # Must always be next_layer.shape[-1]
                                 attn_axes=(1,2,3))(x)   # If input is (Batch, time, d1,...,dn, embed_dim), must be indices of (d1,...,dn)

    pre_decoder_shape = x.shape

    x = Reshape((y_data_shape[1], pre_decoder_shape[1], pre_decoder_shape[2], pre_decoder_shape[3]))(x)
    #----------------------------------------------------------------
    ''' Decoder '''
    
    # apply a single CONV_TRANSPOSE layer used to recover the
    # original depth of the image
    x = Conv3DTranspose(filters=num_filters, # On website 32
                        kernel_size=(y_data_shape[1], 5, 5),
                        strides=y_strides,
                        padding="same",
                        activation='linear')(x)

    x = LayerNormalization()(x)
    
    output = Conv3DTranspose(filters=y_data_shape[4], 
                             kernel_size=(y_data_shape[1], 3, 3),
                             strides=(y_data_shape[1], 1, 1),
                             padding="same",
                             activation='linear')(x)
    #----------------------------------------------------------------
    model = keras.Model(input_, output, name='ConvLSTM')

    # keras.utils.plot_model(model, show_shapes=True, to_file=os.path.join('SavedModels/Figs', 'ConvLSTM.png'))
    # print(model.summary())
    return model

#====================================================================
# x = np.random.random((200, 5, 28, 14, 8))
# y = np.random.random((200, 1, 28, 14, 1)) * 1.2

# model = MakeConvLSTM('config.yml', x.shape, y.shape)

# model.compile(loss=keras.losses.MeanSquaredError(reduction="sum_over_batch_size", 
#                                                  name="MSE"),
#               optimizer=keras.optimizers.Adam(learning_rate=1e-3))

# history = model.fit(x=x,
#                     y=y,
#                     batch_size=10,
#                     epochs=10,
#                     verbose=1)

