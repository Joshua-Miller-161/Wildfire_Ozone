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
total_epochs = 10000
init_lr = 0.01
floor_lr = 0.00001
period = 25
num_waves = 2

# def MajorPeakHeight(epoch, init_lr, floor_lr, total_epochs, period, num_waves):
#     m = (floor_lr - init_lr) / total_epochs
#     x = (num_waves * period) * int(epoch / (num_waves * period))
#     return m * x + init_lr

def MajorPeakHeight(epoch, init_lr, floor_lr, total_epochs, period, num_waves):
    n = int(epoch / (num_waves * period))

    m = - (1 / ((n + 1) * (n + 2))) * (floor_lr + init_lr) / (num_waves * period)

    b = (floor_lr + init_lr) / (n + 1) + (n * (floor_lr + init_lr)) / ((n + 1) * (n + 2))

    x = (num_waves * period) * int(epoch / (num_waves * period))
    return m * x + b

def SubPeakHeight(epoch, top, floor_lr, period, num_waves):
    m = (floor_lr - top) / (period * num_waves)
    y_curr = m * epoch + top
    y_peak = m * (period * int(epoch / period)) + top
    return max(y_curr, y_peak)

def WaveLine(epoch, peak, floor_lr, period):
    m = (floor_lr - peak) / period
    b = peak - m * period * (int(epoch / period))
    return m * epoch + b

epochs = np.arange(total_epochs)
lrs    = []
majors = []

for epoch in range(total_epochs):
    #start_lr = ((floor_lr - init_lr) / total_epochs) * (num_waves * period) * int(epoch / (num_waves * period)) + init_lr
    major   = MajorPeakHeight(epoch, init_lr, floor_lr, total_epochs, period, num_waves) #(num_waves * period)
    peak_lr = SubPeakHeight(epoch % (num_waves * period), major, floor_lr, period, num_waves)
    lr      = WaveLine(epoch, peak_lr, floor_lr, period)
    
    lrs.append(lr)
    majors.append(major)

    print("epoch=", epoch, ", major=", major, ", peak_lr=", peak_lr, ", lr=", lr)

#print(lrs)

plt.plot(epochs, lrs)
plt.plot(epochs, majors)
plt.ylim(floor_lr, init_lr)
plt.show()