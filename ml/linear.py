import sys
sys.dont_write_bytecode = True
import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import netCDF4 as nc
import yaml

from tensorflow import keras
from keras.layers import Input, Concatenate, Dense, TimeDistributed, Reshape, Flatten

sys.path.append(os.getcwd())
from ml.ml_utils import Funnel

#====================================================================
def MakeLinear(config_path, 
               x_data_shape=(1164, 5, 28, 14, 8), 
               y_data_shape=(1164, 1, 28, 14, 1), 
               to_file=None):
    #----------------------------------------------------------------
    ''' Setup '''
    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    assert (config['MODEL_TYPE'] == 'Linear'), "'MODEL_TYPE' must be 'Linear'. Got: "+str(config['MODEL_TYPE'])
    #----------------------------------------------------------------
    input_layer = Input(shape=x_data_shape[1:])

    x = Flatten()(input_layer)

    x = Dense(units=y_data_shape[2] * y_data_shape[3] * y_data_shape[4], activation='linear')(x)

    output_layer = Reshape(y_data_shape[1:])(x)
    #----------------------------------------------------------------
    model = keras.Model(input_layer, output_layer)

    keras.utils.plot_model(model, show_shapes=True, to_file=os.path.join('SavedModels/Figs', 'linear.png'))
    print(model.summary())
    return model

#====================================================================
# x = np.random.random((200, 5, 28, 14, 8))
# y = np.random.random((200, 1, 28, 14, 1)) * 1.2

# model = MakeLinear('config.yml', x.shape, y.shape)

# model.compile(loss=keras.losses.MeanSquaredError(reduction="sum_over_batch_size", 
#                                                  name="MSE"),
#               optimizer=keras.optimizers.Adam(learning_rate=1e-3))

# history = model.fit(x=x,
#                     y=y,
#                     batch_size=10,
#                     epochs=10,
#                     verbose=1)