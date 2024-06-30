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
from keras.layers import Input, Concatenate, TimeDistributed, Dense, LSTM, LayerNormalization, Dropout, AveragePooling3D, UpSampling3D, Reshape, Flatten, LeakyReLU

sys.path.append(os.getcwd())
from ml.ml_utils import Funnel

#====================================================================
def MakeLSTM(config_path, 
             x_data_shape=(1164, 5, 28, 14, 8), 
             y_data_shape=(1164, 1, 28, 14, 1), 
             to_file=None):
    #----------------------------------------------------------------
    ''' Setup '''
    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    assert (config['MODEL_TYPE'] == 'LSTM'), "'MODEL_TYPE' must be 'LSTM'. Got: "+str(config['MODEL_TYPE'])
    #----------------------------------------------------------------
    input_layer = Input(shape=x_data_shape[1:])

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Outer lstms
    x = Reshape(target_shape=(x_data_shape[1], np.prod(x_data_shape[2:])))(input_layer)
    
    x = LSTM(units=int(x.shape[-1] / 4),
               return_sequences=False)(x)
    x = LayerNormalization()(x)
    x = Dropout(rate=0.1)(x)
   
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Final output
    x = Dense(units=np.prod(y_data_shape[2:]),
                    activation='linear')(x)
    
    output_layer = Reshape(y_data_shape[1:])(x)
    #----------------------------------------------------------------
    model = keras.Model(input_layer, output_layer)

    keras.utils.plot_model(model, show_shapes=True, show_layer_activations=True, to_file=os.path.join('SavedModels/Figs', 'LSTM.png'))
    print(model.summary())
    return model

#====================================================================
# x = np.random.random((200, 5, 80, 80, 8))
# y = np.random.random((200, 1, 80, 80, 1))

# model = MakeLSTM('config.yml', x.shape, y.shape)

# model.compile(loss=keras.losses.MeanSquaredError(reduction="sum_over_batch_size", 
#                                                  name="MSE"),
#               optimizer=keras.optimizers.Adam(learning_rate=1e-3))

# history = model.fit(x=x,
#                     y=y,
#                     batch_size=10,
#                     epochs=10,
#                     verbose=1)