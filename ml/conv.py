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
from keras.layers import Input, Concatenate, TimeDistributed, Dense, Conv2D, Conv3D, ConvLSTM2D, LayerNormalization, Dropout, AveragePooling3D, UpSampling3D, Reshape, Flatten, LeakyReLU

sys.path.append(os.getcwd())
from ml.ml_utils import Funnel

#====================================================================
def MakeConv(config_path, 
             x_data_shape=(1164, 5, 28, 14, 8), 
             y_data_shape=(1164, 1, 28, 14, 1), 
             to_file=None):
    #----------------------------------------------------------------
    ''' Setup '''
    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    outer_filters = config['HYPERPARAMETERS']['conv_hyperparams_dict']['num_outer_filters']
    inner_filters = config['HYPERPARAMETERS']['conv_hyperparams_dict']['num_inner_filters']

    assert (config['MODEL_TYPE'] == 'Conv'), "'MODEL_TYPE' must be 'Conv'. Got: "+str(config['MODEL_TYPE'])
    #----------------------------------------------------------------
    input_layer = Input(shape=x_data_shape[1:])

    # Outer convlstms
    x = Conv3D(filters=outer_filters,
               kernel_size=(6, 6, 6),
               padding="same",
               activation=LeakyReLU(alpha=0.1))(input_layer)
    x = LayerNormalization()(x)
    x = Dropout(rate=0.2)(x)

    x = Conv3D(filters=outer_filters,
               kernel_size=(3, 3, 3),
               padding="same",
               activation=LeakyReLU(alpha=0.1))(x)
    x = LayerNormalization()(x)
    outer_convlstm = Dropout(rate=0.2, name='outer_output')(x)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Middle convlstms
    
    y = AveragePooling3D(pool_size=(2,2,1), strides=(2,2,1))(outer_convlstm)

    y = Conv3D(filters=inner_filters,
                   kernel_size=(3, 3, 3),
                   padding="same",
                   activation=LeakyReLU(alpha=0.1))(y)
    y = LayerNormalization()(y)
    y = Dropout(rate=0.05)(y)

    y = Conv3D(filters=inner_filters,
                   kernel_size=(3, 3, 3),
                   padding="same",
                   activation=LeakyReLU(alpha=0.1))(y)
    y = LayerNormalization()(y)
    middle_convlstm = Dropout(rate=0.05)(y)

    y = Conv3D(filters=outer_filters,
                   kernel_size=(3, 3, 3),
                   padding="same",
                   activation=LeakyReLU(alpha=0.1))(y)
    y = LayerNormalization()(y)
    middle_convlstm = Dropout(rate=0.05, name='middle_output')(y)

    print("MIDDlE", middle_convlstm.shape)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Upscale and join to outer

    up = UpSampling3D(size=(2, 2, 1), name='upsample_to_outer')(middle_convlstm)

    xx = Concatenate(name='join_middle_to_outer')([outer_convlstm, up])

    xx = Reshape((xx.shape[1], xx.shape[2], -1))(xx)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Slim down

    final_filters = Funnel(xx.shape[-1], y_data_shape[4]*16, r=16) # Orig. *32, r=np.e, kernel: (3, 3)
    
    for i in range(len(final_filters)):
        if (i % 2 == 0):
            xx = Conv2D(filters=final_filters[i],
                        kernel_size=(1, 1),
                        padding="same",
                        activation=LeakyReLU(alpha=0.1))(xx)
            xx = LayerNormalization()(xx)
            xx = Dropout(rate=0.05)(xx)
        else:
            xx = Conv2D(filters=final_filters[i],
                        kernel_size=(1, 1),
                        padding="same",
                        activation='sigmoid')(xx)
            xx = LayerNormalization()(xx)
            xx = Dropout(rate=0.05)(xx)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Final output

    xx = Conv2D(filters=y_data_shape[4],
                kernel_size=(1, 1),
                padding="same",
                activation='linear')(xx)
    
    output_layer = Reshape(y_data_shape[1:])(xx)
    #----------------------------------------------------------------
    model = keras.Model(input_layer, output_layer)

    keras.utils.plot_model(model, show_shapes=True, show_layer_activations=True, to_file=os.path.join('SavedModels/Figs', 'Conv.png'))
    print(model.summary())
    return model

#====================================================================
# x = np.random.random((200, 28, 14, 5, 8))
# y = np.random.random((200, 28, 14, 1, 1)) * 1.2

# model = MakeConv('config.yml', x.shape, y.shape)

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