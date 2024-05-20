import sys
sys.dont_write_bytecode = True
import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import netCDF4 as nc

import keras_tuner as kt
from tensorflow import keras
from keras.layers import Input, Dense, Conv2D, Flatten, LayerNormalization, Dropout

sys.path.append(os.getcwd())
from ml.ml_utils import Funnel
#====================================================================
hidden_size = 5000
out_size = 5
sizes = Funnel(hidden_size, out_size)

x = np.random.random((100, 10, 10, 1))
y = np.random.random((100, 5))

input  = Input(x.shape[1:])
conv   = Conv2D(3, kernel_size=(2,2), activation='gelu')(input)
flat   = Flatten()(conv)
flat   = LayerNormalization()(flat)
hidden = Dense(5000, 'relu')(flat)

for i in range(len(sizes)):
    dense  = Dense(units=sizes[i], activation='relu')(hidden)
    norm   = LayerNormalization()(dense)
    hidden = Dropout(.1)(norm)

output = Dense(5, 'relu')(hidden)

model  = keras.Model(input, output)
keras.utils.plot_model(model, show_shapes=True, to_file=os.path.join('SavedModels/Figs', 'lol.png'))
print(model.summary())
plt.show()
model.compile(loss=keras.losses.MeanSquaredError(reduction="sum_over_batch_size", 
                                                 name="MSE"),
              optimizer=keras.optimizers.Adam(learning_rate=1e-3))

history = model.fit(x=x,
                    y=y,
                    batch_size=10,
                    epochs=10,
                    verbose=1)