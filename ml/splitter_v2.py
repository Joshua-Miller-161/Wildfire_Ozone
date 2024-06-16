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
#====================================================================
def MakeSplitter_v2(config_path, 
                    x_data_shape=(1164, 5, 28, 14, 8), 
                    y_data_shape=(1164, 1, 28, 14, 1), 
                    to_file=None):
    #----------------------------------------------------------------
    ''' Setup '''
    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    presplit_filters = config['HYPERPARAMETERS']['split_hyperparams_dict']['presplit_filters']
    outer_filters    = config['HYPERPARAMETERS']['split_hyperparams_dict']['num_outer_filters']
    hidden_size      = config['HYPERPARAMETERS']['split_hyperparams_dict']['hidden_size']

    assert (config['MODEL_TYPE'] == 'Split'), "'MODEL_TYPE' must be 'Split'. Got: "+str(config['MODEL_TYPE'])

    lat_kern = -9
    lon_kern = 9
    if (config['REGION'] == 'Whole_Area'):
        lat_kern = 8
        lon_kern = 8
    else:
        lat_kern = 4
        lon_kern = 2
    #----------------------------------------------------------------
    input_layer = Input(shape=x_data_shape[1:])

    # Outer convlstms
    x = Conv3D(filters=outer_filters,
               kernel_size=(x_data_shape[1], lat_kern, lon_kern),
               strides=(1, 1, 1),
               padding="same",
               activation=LeakyReLU(alpha=0.1),
               name='outer')(input_layer)
    x = LayerNormalization()(x) #BatchNormalization()(x)
    outer = Dropout(rate=0.2)(x)

    #final_outer = AveragePooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x) 
    # CONV INSTEAD OF POOL?
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Split

    lat_offset = int(x_data_shape[2] / lat_kern)
    lon_offset = int(x_data_shape[3] / lon_kern)
    print(x_data_shape[2], lat_kern, x_data_shape[3], lon_kern, lon_offset)

    AA = []
    for i in range(0, x_data_shape[2], lat_offset):
        split_outputs = []
        for j in range(0, x_data_shape[3], lon_offset):

            spat_subset = outer[:, :, i:i+lat_offset, j:j+lon_offset, :]

            slim = Conv3D(filters=16,
                          kernel_size=(x_data_shape[1], lat_offset, lon_offset),
                          strides=(1, 1, 1),
                          padding='same',
                          activation=LeakyReLU(alpha=0.1))(spat_subset)
            slim = LayerNormalization()(slim) #BatchNormalization()(x)
            slim = Dropout(rate=0.2)(slim)

            conv = Conv3D(filters=lat_offset * lon_offset * y_data_shape[4],
                          kernel_size=(x_data_shape[1], lat_offset, lon_offset),
                          strides=(x_data_shape[1], lat_offset, lon_offset),
                          padding='same',
                          activation=LeakyReLU(alpha=0.1))(slim)
            conv = LayerNormalization()(conv) #BatchNormalization()(x)
            conv = Dropout(rate=0.2)(conv)
    
            reshape = Reshape(target_shape=(y_data_shape[1], lat_offset, lon_offset, y_data_shape[4]))(conv)

            split_outputs.append(reshape)

        AA.append(split_outputs)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Recombine
    
    final = RecombineLayer(lat_kern=lat_kern)(AA)
    #----------------------------------------------------------------
    model = keras.Model(input_layer, final)

    return model
#====================================================================
# x = np.random.random((200, 5, 28, 14, 8))
# y = np.random.random((200, 1, 28, 14, 1)) * 1.2

# model = MakeSplitter('config.yml', x.shape, y.shape)

# model.compile(loss=keras.losses.MeanSquaredError(reduction="sum_over_batch_size", 
#                                                  name="MSE"),
#               optimizer=keras.optimizers.Adam(learning_rate=1e-3))

# history = model.fit(x=x,
#                     y=y,
#                     batch_size=10,
#                     epochs=2,
#                     verbose=1)

# class NonSharedWeightsConv3D(keras.layers.Layer):
#     def __init__(self, num_filters, kernel_size, **kwargs):
#         super(NonSharedWeightsConv3D, self).__init__(**kwargs)
#         self.num_filters = num_filters
#         self.kernel_size = kernel_size
#         self.conv_layers = [Conv2D(num_filters, kernel_size, padding='same') for _ in range(10)]

#     def call(self, inputs):
#         # Expecting input shape: (batch, time, height, width, channels)
#         outputs = []
#         for i in range(10):
#             # Select the frame at time step i
#             frame = inputs[:, i, :, :, :]
#             # Apply the corresponding Conv2D layer
#             conv_output = self.conv_layersi
#             # Add the conv_output to the list of outputs
#             outputs.append(conv_output)
#         # Stack the outputs along the time dimension
#         return tf.stack(outputs, axis=1)

# # Define the model
# input_shape = (10, 25, 25, 3)
# model = keras.Sequential([
#     NonSharedWeightsConv3D(num_filters=32, kernel_size=(3, 3), input_shape=input_shape),
#     # Additional layers can be added here
#     Reshape((10, 25 * 25 * 32)),  # Reshape for Dense layer
#     Dense(units=25 * 25 * 3),  # Output layer
#     Reshape((1, 25, 25, 3))  # Reshape to the desired output shape
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='mse')

# # Generate synthetic data
# x_train = np.random.rand(100, 10, 25, 25, 3)
# y_train = np.random.rand(100, 1, 25, 25, 3)

# # Train the model
# model.fit(x_train, y_train, epochs=10)