import sys
sys.dont_write_bytecode = True
import numpy as np
import os
import yaml
import pandas as pd
from tensorflow import keras
from keras.callbacks import Callback
#====================================================================
def NameModel(config_path):
    model_name = ''
    print('IN NAME MODEL', model_name)
    #----------------------------------------------------------------
    ''' Get infor from config '''
    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    region       = config['REGION']
    history_vars = config['HISTORY_DATA']
    target_vars  = config['TARGET_DATA']
    rf_offset    = config['RF_OFFSET']
    history_len  = config['HIST_TARG']['history_len']
    target_len   = config['HIST_TARG']['target_len']
    num_trees    = config['HYPERPARAMETERS']['rf_hyperparams_dict']['num_trees']
    num_nuerons  = config['HYPERPARAMETERS']['trans_hyperparams_dict']['num_neurons']
    #----------------------------------------------------------------
    ''' uh '''
    shorthand_dict = {'ozone' : 'O', 
                      'fire'  : 'F',
                      'temp'  : 'T',
                      'u-wind': 'U',
                      'v-wind': 'V',
                      'lon'   : 'X',
                      'lat'   : 'Y',
                      'time'  : 'D',
                      'Whole_Area': 'WA',
                      'North_Land': 'NL',
                      'South_Land': 'SL',
                      'East_Ocean': 'EO',
                      'West_Ocean': 'WO',}
    #----------------------------------------------------------------
    ''' Make names '''

    if (config['MODEL_TYPE'] == 'RF'):
        model_name = 'RF_reg='+shorthand_dict[region]+'_f='+str(int(rf_offset))+'_In='
        #print("model_name:", model_name)

        for var in history_vars:
            model_name += shorthand_dict[var]

        #print("model_name:", model_name)

        model_name += '_Out='
        for var in target_vars:
            model_name += shorthand_dict[var]
        
        model_name += '.joblib'
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif (config['MODEL_TYPE'] == 'Linear'):
        model_name = 'Linear_reg='+shorthand_dict[region]+'_f='+str(int(rf_offset))+'_In='
        #print("model_name:", model_name)

        epochs = config['HYPERPARAMETERS']['linear_hyperparams_dict']['epochs']

        for var in history_vars:
            model_name += shorthand_dict[var]

        model_name += '_Out='
        for var in target_vars:
            model_name += shorthand_dict[var]
        
        model_name += '_e='+str(epochs)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif (config['MODEL_TYPE'] == 'Dense'):
        model_name = 'Dense_reg='+shorthand_dict[region]+'_f='+str(int(rf_offset))+'_In='
        #print("model_name:", model_name)
        epochs = config['HYPERPARAMETERS']['dense_hyperparams_dict']['epochs']

        for var in history_vars:
            model_name += shorthand_dict[var]

        model_name += '_Out='
        for var in target_vars:
            model_name += shorthand_dict[var]
        
        model_name += '_e='+str(epochs)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif (config['MODEL_TYPE'] == 'ConvLSTM'):
        model_name = 'ConvLSTM_reg='+shorthand_dict[region]+'_f='+str(int(rf_offset))+'_In='
        print("model_name:", model_name)
        epochs = config['HYPERPARAMETERS']['convlstm_hyperparams_dict']['epochs']

        for var in history_vars:
            model_name += shorthand_dict[var]

        model_name += '_Out='
        for var in target_vars:
            model_name += shorthand_dict[var]
        
        model_name += '_e='+str(epochs)

    return model_name
#====================================================================
def ParseModelName(input_string, substrs=['reg=', 'In=', 'Out=', 'e='], split_char='_'):

    dict_ = {'WA': 'Whole Area',
             'EO': 'East Ocean',
             'WO': 'West Ocean', 
             'SL': 'South Land',
             'NL': 'North Land',
             'RF': 'Random Forest',
             'Trans': 'Transformer'}

    info = []
    start = len(input_string.split(split_char)[0]) + 1
    try:
        info.append(dict_[input_string.split(split_char)[0]])
    except KeyError:
        info.append(input_string.split(split_char)[0])

    print("info:", info, input_string)

    for substr in substrs:
        temp_str = input_string[start:]
        if (substr in temp_str):

            lol = temp_str.split(split_char)[0]
            lol = lol[len(substr):]
            start += len(substr) + len(lol) + 1
            print(temp_str, substr, lol)

            try:
                info.append(dict_[lol])
            except KeyError:
                info.append(lol)

    return info
#====================================================================
def Funnel(start_size, end_size, r=np.e):
    assert not start_size == end_size, "'start_size' and 'end_size' must be different. Got start_size = {}, output_size = {}".format(start_size, end_size)
    if (start_size > end_size):
        if (round(start_size / r) <= end_size):
            return [start_size, end_size]

        else:
            sizes = [start_size]
            i = 1
            while ((round(start_size / r**i)) > end_size):
                sizes.append(round(start_size / r**i))
                i += 1
            sizes.append(end_size)
            return sizes
    
    elif (start_size < end_size):
        if (start_size * r > end_size):
            return [start_size, end_size]
        
        else:
            sizes = [start_size]
            i = 1
            while ((round(start_size * r**i)) < end_size):
                sizes.append(round(start_size * r**i))
                i += 1
            sizes.append(end_size)
            return sizes
#====================================================================
class TriangleWaveLR(Callback):
    def __init__(self, initial_peak_lr=0.01, floor_lr=0.00001, period=10):
        super().__init__()
        self.initial_peak_lr = initial_peak_lr
        self.peak_lr = initial_peak_lr
        self.floor_lr = floor_lr
        self.period = period

    def on_epoch_begin(self, epoch, logs=None):
        cycle = np.floor(1 + epoch / (2 * self.period))
        x = np.abs(epoch / self.period - 2 * cycle + 1)
        lr = self.floor_lr + (self.peak_lr - self.floor_lr) * np.maximum(0, (1 - x))
        keras.backend.set_value(self.model.optimizer.lr, lr)
        #print(f"Epoch {epoch + 1}: Learning rate = {lr:.6f}")

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            self.initial_peak_lr /= (int((epoch + 1) / self.period) + 1)

#====================================================================
class NoisyDecayLR(Callback):
    def __init__(self, initial_lr, final_lr, total_epochs, mag_noise):
        super().__init__()
        self.initial_lr   = initial_lr
        self.final_lr     = final_lr
        self.total_epochs = total_epochs
        self.mag_noise    = mag_noise

    def on_epoch_begin(self, epoch, logs=None):
        decay_rate = self.final_lr / self.initial_lr
        decayed_lr = self.initial_lr * (decay_rate ** (epoch / self.total_epochs))

        # Add random noise
        noise = np.random.uniform(-1, 1) * self.mag_noise * np.sqrt(decayed_lr * self.initial_lr)
        noisy_lr = decayed_lr + noise

        # Set the learning rate for this epoch
        keras.backend.set_value(self.model.optimizer.lr, noisy_lr)
        #print(f"Epoch {epoch + 1} - Learning rate: {noisy_lr:.6f}")