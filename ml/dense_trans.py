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
from keras.layers import Input, Concatenate, Dense, TimeDistributed, Permute, Reshape, Flatten, LayerNormalization, Dropout

sys.path.append(os.getcwd())
from ml.ml_utils import Funnel
from ml.custom_keras_layers import Transpose, TransformerBlock
#====================================================================
def MakeDenseTrans(config_path, 
              x_data_shape=(1164, 5, 28, 14, 8), 
              y_data_shape=(1164, 1, 28, 14, 1), 
              to_file=None):
    #----------------------------------------------------------------
    ''' Setup '''
    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    assert (config['MODEL_TYPE'] == 'Trans'), "'MODEL_TYPE' must be 'Dense'. Got: "+str(config['MODEL_TYPE'])
    
    num_heads = config['HYPERPARAMETERS']['trans_hyperparams_dict']['num_heads']
    num_trans = 2
    #----------------------------------------------------------------
    input_layer = Input(shape=x_data_shape[1:])

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Spatial part
    spat_flat = Reshape((input_layer.shape[1], -1))(input_layer)
    
    num_spat_neurons = Funnel(spat_flat.shape[-1], y_data_shape[2] * y_data_shape[3] * y_data_shape[4], r=10)
    for i in range(1, np.shape(num_spat_neurons)[0]):
        spat_flat = TransformerBlock(embed_dim=spat_flat.shape[-1],
                                     num_heads=num_heads,
                                     ff_dim=spat_flat.shape[-1])(spat_flat)
        spat_flat = TimeDistributed(Dense(units=num_spat_neurons[i], activation='relu'))(spat_flat)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Time part
    time = Permute((2, 3, 1, 4), input_shape=input_layer.shape[1:])(input_layer)
    time_flat = Reshape((time.shape[1]*time.shape[2], time.shape[3]*time.shape[4]))(time)
    
    num_time_neurons = Funnel(time_flat.shape[-1], x_data_shape[1], r=10)
    for i in range(1, np.shape(num_time_neurons)[0]):
        time_flat = TransformerBlock(embed_dim=time_flat.shape[-1],
                                     num_heads=num_heads,
                                     ff_dim=num_time_neurons[i])(time_flat)
        time_flat = TimeDistributed(Dense(units=num_time_neurons[i], activation='relu'))(time_flat)
    # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # # Rejoin
    time_flat = Permute((2, 1))(time_flat)
    final_dense = Concatenate()([time_flat, spat_flat])

    flat = Flatten()(final_dense)

    num_flat_neurons = Funnel(flat.shape[-1], int(y_data_shape[2] * y_data_shape[3] * y_data_shape[4] * np.e)+1)
    for i in range(np.shape(num_flat_neurons)[0]):
        flat = Dense(units=num_flat_neurons[i], activation='relu')(flat)
        flat = LayerNormalization()(flat)
        flat = Dropout(rate=0.1)(flat)
    
    final_dense = Dense(units=y_data_shape[2] * y_data_shape[3] * y_data_shape[4], activation='linear')(flat)

    output_layer = Reshape(y_data_shape[1:])(final_dense)
    #----------------------------------------------------------------
    model = keras.Model(input_layer, output_layer)

    keras.utils.plot_model(model, show_shapes=True, show_layer_activations=True, to_file=os.path.join('SavedModels/Figs', 'DiamondDenseTrans.png'))
    print(model.summary())
    return model

#====================================================================
# x = np.random.random((200, 5, 28, 14, 8))
# y = np.random.random((200, 1, 28, 14, 1)) * 1.2

# model = MakeDenseTrans('config.yml', x.shape, y.shape)

# model.compile(loss=keras.losses.MeanSquaredError(reduction="sum_over_batch_size", 
#                                                  name="MSE"),
#               optimizer=keras.optimizers.Adam(learning_rate=1e-3))

# history = model.fit(x=x,
#                     y=y,
#                     batch_size=10,
#                     epochs=10,
#                     verbose=1)