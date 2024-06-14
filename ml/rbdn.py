import sys
sys.dont_write_bytecode = True
import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import netCDF4 as nc
import yaml

import keras_tuner as kt
from tensorflow import keras
from keras.layers import Input, Concatenate, TimeDistributed, Dense, Conv2D, Conv3D, ConvLSTM2D, LayerNormalization, BatchNormalization, Dropout, AveragePooling3D, UpSampling3D, Reshape, Flatten, LeakyReLU, Convolution2DTranspose, Convolution3DTranspose

sys.path.append(os.getcwd())
from ml.ml_utils import Funnel

#====================================================================
# def MakeRBDN(config_path, 
#              x_data_shape=(1164, 5, 28, 14, 8), 
#              y_data_shape=(1164, 1, 28, 14, 1), 
#              to_file=None):
#     #----------------------------------------------------------------
#     ''' Setup '''
#     with open(config_path, 'r') as c:
#         config = yaml.load(c, Loader=yaml.FullLoader)

#     outer_filters = config['HYPERPARAMETERS']['conv_hyperparams_dict']['num_outer_filters']
#     inner_filters = config['HYPERPARAMETERS']['conv_hyperparams_dict']['num_inner_filters']

#     assert (config['MODEL_TYPE'] == 'RBDN'), "'MODEL_TYPE' must be 'RBDN'. Got: "+str(config['MODEL_TYPE'])

#     lat_kern = -9
#     lon_kern = 9
#     if (config['REGION'] == 'Whole_Area'):
#         lat_kern = 8
#         lon_kern = 8
#     else:
#         lat_kern = 4
#         lon_kern = 2
#     #----------------------------------------------------------------
#     input_layer = Input(shape=x_data_shape[1:])

#     # Outer convlstms
#     x = Conv3D(filters=outer_filters,
#                kernel_size=(1, lat_kern, lon_kern),
#                strides=(1, lat_kern, lon_kern),
#                padding="same",
#                activation=LeakyReLU(alpha=0.1))(input_layer)
#     x = LayerNormalization()(x) #BatchNormalization()(x)
#     final_outer = Dropout(rate=0.2)(x)

#     x = AveragePooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(final_outer) 
#     # CONV INSTEAD OF POOL?
#     # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#     # Middle convlstms
    
#     y = Conv3D(filters=inner_filters,
#                kernel_size=(int(x.shape[1] / 2), 3, 3),
#                strides=(int(x.shape[1] / 2), 3, 3),
#                padding="same",
#                activation=LeakyReLU(alpha=0.1))(x)
#     y = LayerNormalization()(y) #BatchNormalization()(y)
#     y = Dropout(rate=0.05)(y)

#     y = Conv3D(filters=outer_filters,
#                kernel_size=(y.shape[1], 3, 3),
#                strides=(y.shape[1], 3, 3),
#                padding="same",
#                activation=LeakyReLU(alpha=0.1))(y)
#     y = LayerNormalization()(y) #BatchNormalization()(y)
#     y = Dropout(rate=0.05)(y)
    
#     #y = AveragePooling3D(pool_size=(1, 1, 1), strides=(1, 1, 1))(y)
#     # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#     # Final output
#     z = UpSampling3D(size=(1, final_outer.shape[2], final_outer.shape[3]))(y)
#     output_layer = Convolution3DTranspose(filters=y_data_shape[4],
#                                kernel_size=(y_data_shape[1], int(y_data_shape[2] / final_outer.shape[2]), int(y_data_shape[3] / final_outer.shape[3])),
#                                strides=(y_data_shape[1], int(y_data_shape[2] / final_outer.shape[2]), int(y_data_shape[3] / final_outer.shape[3])),
#                                padding="same",
#                                activation='linear')(z)
#     #----------------------------------------------------------------
#     model = keras.Model(input_layer, output_layer)

#     keras.utils.plot_model(model, show_shapes=True, show_layer_activations=True, to_file=os.path.join('SavedModels/Figs', 'RBDN.png'))
#     print(model.summary())
#     return model

#====================================================================
def MakeRBDN(config_path, 
             x_data_shape=(1164, 5, 28, 14, 8), 
             y_data_shape=(1164, 1, 28, 14, 1), 
             to_file=None):
    #----------------------------------------------------------------
    ''' Setup '''
    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    outer_filters = config['HYPERPARAMETERS']['rbdn_hyperparams_dict']['num_outer_filters']
    inner_filters = config['HYPERPARAMETERS']['rbdn_hyperparams_dict']['num_inner_filters']

    assert (config['MODEL_TYPE'] == 'RBDN'), "'MODEL_TYPE' must be 'RBDN'. Got: "+str(config['MODEL_TYPE'])

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
               kernel_size=(1, lat_kern, lon_kern),
               strides=(1, 1, 1),
               padding="same",
               activation=LeakyReLU(alpha=0.1))(input_layer)
    x = LayerNormalization()(x) #BatchNormalization()(x)
    x = Dropout(rate=0.2)(x)

    final_outer = AveragePooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x) 
    # CONV INSTEAD OF POOL?
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Middle convlstms
    
    y = Conv3D(filters=inner_filters,
               kernel_size=(int(x.shape[1] / 2), 3, 3),
               strides=(int(x.shape[1] / 2), 1, 1),
               padding="same",
               activation=LeakyReLU(alpha=0.1))(final_outer)
    y = LayerNormalization()(y) #BatchNormalization()(y)
    y = Dropout(rate=0.05)(y)

    y = Conv3D(filters=outer_filters,
               kernel_size=(y.shape[1], 3, 3),
               strides=(y.shape[1], 1, 1),
               padding="same",
               activation=LeakyReLU(alpha=0.1))(y)
    y = LayerNormalization()(y) #BatchNormalization()(y)
    y = Dropout(rate=0.05)(y)
    
    #y = AveragePooling3D(pool_size=(1, 1, 1), strides=(1, 1, 1))(y)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Final output
    z = UpSampling3D(size=(1, int(y_data_shape[2] / final_outer.shape[2]), int(y_data_shape[3] / final_outer.shape[3])))(y)
    output_layer = Convolution3DTranspose(filters=y_data_shape[4],
                               kernel_size=(y_data_shape[1], int(y_data_shape[2] / final_outer.shape[2]), int(y_data_shape[3] / final_outer.shape[3])),
                               strides=(1, 1, 1),
                               padding="same",
                               activation='linear')(z)
    #----------------------------------------------------------------
    model = keras.Model(input_layer, output_layer)

    keras.utils.plot_model(model, show_shapes=True, show_layer_activations=True, to_file=os.path.join('SavedModels/Figs', 'RBDN.png'))
    print(model.summary())
    return model
#====================================================================
# x = np.random.random((200, 5, 28, 14, 8))
# y = np.random.random((200, 1, 28, 14, 1)) * 1.2

# model = MakeRBDN('config.yml', x.shape, y.shape)

# model.compile(loss=keras.losses.MeanSquaredError(reduction="sum_over_batch_size", 
#                                                  name="MSE"),
#               optimizer=keras.optimizers.Adam(learning_rate=1e-3))

# history = model.fit(x=x,
#                     y=y,
#                     batch_size=10,
#                     epochs=10,
#                     verbose=1)

# data = np.random.random(size=(10, 5, 5, 3))

# reshaped_data = np.transpose(data, (1, 2, 0, 3))

# idx = 1

# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(data[idx, :, :, :])
# ax[1].imshow(reshaped_data[:, :, idx, :])

# plt.show()