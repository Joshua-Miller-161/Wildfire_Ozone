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
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Concatenate, Reshape, Permute
#====================================================================
''' Setup '''
x_data = tf.random.uniform([1, 1, 28, 14, 1], minval=0, maxval=1)

lat_kern = 4
lon_kern = 2
#----------------------------------------------------------------
input_layer = Input(shape=x_data.shape[1:])

split_outputs = []
lat_offset = int(x_data.shape[2] / lat_kern)
lon_offset = int(x_data.shape[3] / lon_kern)

AA = []
for i in range(0, x_data.shape[2], lat_offset):
    split_outputs = []
    for j in range(0, x_data.shape[3], lon_offset):

        spat_subset = input_layer[:, :, i:i+lat_offset, j:j+lon_offset, :]

        #reshape = Reshape(target_shape=(x_data.shape[1], lat_offset, lon_offset, x_data.shape[4]))(spat_subset)

        split_outputs.append(spat_subset)
    AA.append(split_outputs)

BB = []
for i in range(lat_kern):
    BB.append(Concatenate(axis=-2)(AA[i]))

final = Concatenate(axis=-3)(BB)

# BB = []
# for i in range(lat_kern):
#     BB.append(tf.concat(AA[i], axis=-2))

# final = tf.concat(BB, axis=-3)

# print("   -   -")
# print("AAAAAA", tf.shape(AA))
# print("   -   -")
# print("BBBBBB", tf.shape(BB))
# print("   -   -")
# print("final", tf.shape(final))
# print("   -   -")
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Recombine
# BB = []
# for i in range(0, x_data.shape[2], lat_offset):
#     BB.append(Concatenate(axis=-2)(AA))


# recombined = Concatenate(axis=-3)(recombined)  # Then concatenate along longitude
# recombined = Reshape(target_shape=x_data.shape[1:])(recombined)

model = keras.Model(input_layer, final)
keras.utils.plot_model(model, show_shapes=True, show_layer_activations=True)
print(model.summary())
#====================================================================

x_data = tf.random.uniform([1, 1, 28, 14, 1], minval=0, maxval=1)

y_pred = model.predict(x_data)

print(np.shape(y_pred))
#print(y_pred)

print("============================================================")
print("============================================================")
print("============================================================")
print("============================================================")
print(x_data - y_pred)