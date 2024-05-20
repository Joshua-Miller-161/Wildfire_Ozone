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
from keras.layers import Input, Concatenate, Dense, ConvLSTM2D, LayerNormalization, BatchNormalizationV2, Dropout, AveragePooling3D, UpSampling3D, Reshape

sys.path.append(os.getcwd())
from ml.ml_utils import Funnel

#====================================================================
def MakeConvLSTM(config_path, 
                 x_data_shape=(1164, 5, 28, 14, 8), 
                 y_data_shape=(1164, 1, 28, 14, 1), 
                 to_file=None):
    #----------------------------------------------------------------
    ''' Get data from config '''
    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    outer_filters = config['HYPERPARAMETERS']['convlstm_dict']['num_outer_filters']
    inner_filters = config['HYPERPARAMETERS']['convlstm_dict']['num_inner_filters']
    #----------------------------------------------------------------
    input_layer = Input(shape=x_data_shape[1:])

    # Outer convlstms
    x = ConvLSTM2D(filters=outer_filters,
                   kernel_size=(6, 6),
                   padding="same",
                   return_sequences=True,
                   activation="tanh")(input_layer)
    x = LayerNormalization()(x)
    x = Dropout(rate=0.2)(x)

    x = ConvLSTM2D(filters=outer_filters,
                   kernel_size=(3, 3),
                   padding="same",
                   return_sequences=True,
                   activation="tanh")(x)
    x = LayerNormalization()(x)
    outer_convlstm = Dropout(rate=0.2, name='outer_output')(x)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Middle convlstms
    y = AveragePooling3D(pool_size=(1,2,2), strides=(1,2,2))(outer_convlstm)

    y = ConvLSTM2D(filters=inner_filters,
                   kernel_size=(3, 3),
                   padding="same",
                   return_sequences=True,
                   activation="tanh")(y)
    y = LayerNormalization()(y)
    y = Dropout(rate=0.05)(y)

    y = ConvLSTM2D(filters=inner_filters,
                   kernel_size=(3, 3),
                   padding="same",
                   return_sequences=True,
                   activation="tanh")(y)
    y = LayerNormalization()(y)
    middle_convlstm = Dropout(rate=0.05)(y)

    y = ConvLSTM2D(filters=outer_filters,
                   kernel_size=(3, 3),
                   padding="same",
                   return_sequences=True,
                   activation="tanh")(y)
    y = LayerNormalization()(y)
    middle_convlstm = Dropout(rate=0.05, name='middle_output')(y)

    print("MIDDlE", middle_convlstm.shape)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Upscale and join to outer

    up = UpSampling3D(size=(1, 2, 2), name='upsample_to_outer')(middle_convlstm)

    merge = Concatenate(name='join_middle_to_outer')([outer_convlstm, up])
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Final convlstm outputs

    xx = ConvLSTM2D(filters=8,
                   kernel_size=(3, 3),
                   padding="same",
                   return_sequences=True,
                   activation="tanh")(merge)
    xx = LayerNormalization()(xx)
    xx = Dropout(rate=0.05)(xx)

    xx = ConvLSTM2D(filters=y_data_shape[4],
                    kernel_size=(3, 3),
                    padding="same",
                    return_sequences=False,
                    activation="tanh")(xx)
    xx = Reshape((y_data_shape[1], y_data_shape[2], y_data_shape[3], y_data_shape[4]))(xx)
    xx = LayerNormalization()(xx)
    xx = Dropout(rate=0.05)(xx)
    
    output_layer = Dense(units=y_data_shape[-1], activation='linear')(xx)

    #----------------------------------------------------------------
    model = keras.Model(input_layer, output_layer)

    keras.utils.plot_model(model, show_shapes=True, to_file=os.path.join('SavedModels/Figs', 'lol.png'))
    print(model.summary())
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