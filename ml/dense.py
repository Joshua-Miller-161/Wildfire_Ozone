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
from keras.layers import Input, Concatenate, Dense, TimeDistributed, Reshape, Flatten, LayerNormalization, Dropout

sys.path.append(os.getcwd())
from ml.ml_utils import Funnel

#====================================================================
def MakeDense(config_path, 
              x_data_shape=(1164, 5, 28, 14, 8), 
              y_data_shape=(1164, 1, 28, 14, 1), 
              to_file=None):
    #----------------------------------------------------------------
    ''' Setup '''
    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    assert (config['MODEL_TYPE'] == 'Dense'), "'MODEL_TYPE' must be 'Dense'. Got: "+str(config['MODEL_TYPE'])
    #----------------------------------------------------------------
    input_layer = Input(shape=x_data_shape[1:])

    x = Flatten()(input_layer)

    num_neurons = Funnel(x.shape[-1], int(y_data_shape[2] * y_data_shape[3] * y_data_shape[4] * np.e))
    for i in range(np.shape(num_neurons)[0]):
        x = Dense(units=num_neurons[i], activation='relu')(x)
        x = LayerNormalization()(x)
        x = Dropout(rate=0.1)(x)
    
    x = Dense(units=y_data_shape[2] * y_data_shape[3] * y_data_shape[4], activation='linear')(x)

    output_layer = Reshape(y_data_shape[1:])(x)
    #----------------------------------------------------------------
    model = keras.Model(input_layer, output_layer)

    keras.utils.plot_model(model, show_shapes=True, to_file=os.path.join('SavedModels/Figs', 'dense.png'))
    print(model.summary())
    return model

#====================================================================
# x = np.random.random((200, 5, 28, 14, 8))
# y = np.random.random((200, 1, 28, 14, 1)) * 1.2

# model = MakeDense('config.yml', x.shape, y.shape)

# model.compile(loss=keras.losses.MeanSquaredError(reduction="sum_over_batch_size", 
#                                                  name="MSE"),
#               optimizer=keras.optimizers.Adam(learning_rate=1e-3))

# history = model.fit(x=x,
#                     y=y,
#                     batch_size=10,
#                     epochs=10,
#                     verbose=1)