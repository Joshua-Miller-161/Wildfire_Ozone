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
def MakeSplitter(config_path, 
                 x_data_shape=(1164, 5, 28, 14, 8), 
                 y_data_shape=(1164, 1, 28, 14, 1), 
                 to_file=None):
    #----------------------------------------------------------------
    ''' Setup '''
    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    presplit_filters = config['HYPERPARAMETERS']['split_hyperparams_dict']['presplit_filters']
    outer_filters    = config['HYPERPARAMETERS']['split_hyperparams_dict']['num_outer_filters']
    hidden_size = config['HYPERPARAMETERS']['split_hyperparams_dict']['hidden_size']

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

    keras.utils.plot_model(model, show_shapes=True, show_layer_activations=True, to_file=os.path.join('SavedModels/Figs', 'splitter.png'))
    print(model.summary())
    return model
#====================================================================
x = np.random.random((200, 5, 28, 14, 8))
y = np.random.random((200, 1, 28, 14, 1)) * 1.2

model = MakeSplitter('config.yml', x.shape, y.shape)

model.compile(loss=keras.losses.MeanSquaredError(reduction="sum_over_batch_size", 
                                                 name="MSE"),
              optimizer=keras.optimizers.Adam(learning_rate=1e-3))

history = model.fit(x=x,
                    y=y,
                    batch_size=10,
                    epochs=2,
                    verbose=1)
# #====================================================================
# model_json = model.to_json()
# with open(os.path.join('/Users/joshuamiller/Documents/Lancaster/SavedModels/Split', 'splitter_test.json'), 'w') as json_file:
#     json_file.write(model_json)

# model.save_weights(os.path.join('/Users/joshuamiller/Documents/Lancaster/SavedModels/Split', 'splitter_test.h5'))
# #--------------------------------------------------------------------
# with open(os.path.join('/Users/joshuamiller/Documents/Lancaster/SavedModels/Split', 'splitter_test.json'), 'r') as json_file:
#     loaded_model_json = json_file.read()
# model = keras.models.model_from_json(loaded_model_json, custom_objects={'RecombineLayer': RecombineLayer})

# model.load_weights(os.path.join('/Users/joshuamiller/Documents/Lancaster/SavedModels/Split', 'splitter_test.h5'))
# #--------------------------------------------------------------------
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
#               loss=keras.losses.MeanSquaredError(reduction="sum_over_batch_size", name="MSE"))

# y_pred = model.predict(x)




# data = np.random.random(size=(10, 5, 5, 3))

# reshaped_data = np.transpose(data, (1, 2, 0, 3))

# idx = 1

# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(data[idx, :, :, :])
# ax[1].imshow(reshaped_data[:, :, idx, :])

# plt.show()