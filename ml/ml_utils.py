import sys
sys.dont_write_bytecode = True
import numpy as np
import os
import yaml
import pandas as pd
from tensorflow import keras
from keras.callbacks import Callback
import matplotlib.pyplot as plt
#====================================================================
def NameModel(config_path, prefix=''):
    model_name = ''
    if not (prefix == ''):
        model_name = prefix+'-'
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
    num_neurons  = config['HYPERPARAMETERS']['trans_hyperparams_dict']['num_neurons']
    num_boost_round       = config['HYPERPARAMETERS']['gb_hyperparams_dict']['num_boost_round']
    max_depth             = config['HYPERPARAMETERS']['gb_hyperparams_dict']['max_depth']
    early_stopping_rounds = config['HYPERPARAMETERS']['gb_hyperparams_dict']['early_stopping_rounds']
    subsample             = config['HYPERPARAMETERS']['gb_hyperparams_dict']['subsample']
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
        if (config['HYPERPARAMETERS']['rf_hyperparams_dict']['use_xgboost'] == True):
            model_name += 'XGBRF_reg='+shorthand_dict[region]+'_f='+str(int(rf_offset))+'_In='
        else:
            model_name += 'RF_reg='+shorthand_dict[region]+'_f='+str(int(rf_offset))+'_In='

        for var in history_vars:
            model_name += shorthand_dict[var]

        model_name += '_Out='
        for var in target_vars:
            model_name += shorthand_dict[var]
        
        if (config['HYPERPARAMETERS']['rf_hyperparams_dict']['use_xgboost'] == True):
            model_name += '.pkl'
        else:
            model_name += '.joblib'
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if (config['MODEL_TYPE'] == 'GBM'):
        model_name += 'GBM_reg='+shorthand_dict[region]+'_f='+str(int(rf_offset))+'_In='
        #print("model_name:", model_name)

        for var in history_vars:
            model_name += shorthand_dict[var]

        #print("model_name:", model_name)

        model_name += '_Out='
        for var in target_vars:
            model_name += shorthand_dict[var]
        
        model_name += '.pkl'
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif (config['MODEL_TYPE'] == 'Linear'):
        model_name += 'Linear_reg='+shorthand_dict[region]+'_f='+str(int(rf_offset))+'_In='
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
        model_name += 'Dense_reg='+shorthand_dict[region]+'_h='+str(int(history_len))+'_f='+str(int(target_len))+'_In='
        #print("model_name:", model_name)
        epochs = config['HYPERPARAMETERS']['dense_hyperparams_dict']['epochs']

        for var in history_vars:
            model_name += shorthand_dict[var]

        model_name += '_Out='
        for var in target_vars:
            model_name += shorthand_dict[var]
        
        model_name += '_e='+str(epochs)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif (config['MODEL_TYPE'] == 'Conv'):
        model_name = 'Conv_reg='+shorthand_dict[region]+'_h='+str(int(history_len))+'_f='+str(int(target_len))+'_In='
        print("model_name:", model_name)
        epochs = config['HYPERPARAMETERS']['conv_hyperparams_dict']['epochs']

        for var in history_vars:
            model_name += shorthand_dict[var]

        model_name += '_Out='
        for var in target_vars:
            model_name += shorthand_dict[var]
        
        model_name += '_e='+str(epochs)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif (config['MODEL_TYPE'] == 'ConvLSTM'):
        model_name += 'ConvLSTM_reg='+shorthand_dict[region]+'_h='+str(int(history_len))+'_f='+str(int(target_len))+'_In='
        print("model_name:", model_name)
        epochs = config['HYPERPARAMETERS']['convlstm_hyperparams_dict']['epochs']

        for var in history_vars:
            model_name += shorthand_dict[var]

        model_name += '_Out='
        for var in target_vars:
            model_name += shorthand_dict[var]
        
        model_name += '_e='+str(epochs)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif (config['MODEL_TYPE'] == 'Trans'):
        model_name += 'Trans_reg='+shorthand_dict[region]+'_h='+str(int(history_len))+'_f='+str(int(target_len))+'_In='
        print("model_name:", model_name)
        epochs = config['HYPERPARAMETERS']['trans_hyperparams_dict']['epochs']

        for var in history_vars:
            model_name += shorthand_dict[var]

        model_name += '_Out='
        for var in target_vars:
            model_name += shorthand_dict[var]
        
        model_name += '_e='+str(epochs)

    return model_name
#====================================================================
def ParseModelName(input_string, substrs=['reg=', 'h=', 'f=', 'In=', 'Out=', 'e='], split_char='_'):

    dict_ = {'WA': 'Whole_Area',
             'EO': 'East_Ocean',
             'WO': 'West_Ocean', 
             'SL': 'South_Land',
             'NL': 'North_Land',
             'RF': 'Random Forest',
             'XGBRF': 'XGBoost Random Forest',
             'GBM': 'Gradient Boosting Machine',
             'Trans': 'Transformer',
             'O': 'ozone',
             'F': 'fire',
             'T': 'temp',
             'U': 'u-wind',
             'V': 'v-wind',
             'X': 'lon',
             'Y': 'lat',
             'D': 'time'}
    
    final_dict = {}
    #----------------------------------------------------------------
    input_string = input_string.split('.')[0]

    print('input_string:', input_string)

    chunks = input_string.split(split_char)

    info = []
    try:
        info.append(dict_[chunks[0]])
    except KeyError:
        info.append(chunks[0])

    final_dict['MODEL_TYPE'] = chunks[0]
    final_dict['MODEL_TYPE_LONG'] = info[0]

    for i in range(1, len(chunks)):
        for substr in substrs:
            if (substr in chunks[i]):
                lol = chunks[i][len(substr):]
                try:
                    info.append(dict_[lol])
                except KeyError:
                    info.append(lol)

                if (substr == 'reg='):
                    final_dict['REGION'] = info[-1]
                
                elif (substr == 'h='):
                    final_dict['history_len'] = int(lol)
                
                elif (substr == 'f='):
                    final_dict['target_len'] = int(lol)
                    final_dict['RF_OFFSET']  = int(lol)
                
                elif (substr == 'In='):
                    HISTORY_DATA = []
                    for char in lol:
                        HISTORY_DATA.append(dict_[char])
                    final_dict['HISTORY_DATA'] = HISTORY_DATA
                
                elif (substr == 'Out='):
                    TARGET_DATA = []
                    for char in lol:
                        TARGET_DATA.append(dict_[char])
                    final_dict['TARGET_DATA'] = TARGET_DATA
    #----------------------------------------------------------------
    return info, final_dict
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
    def __init__(self, total_epochs, initial_lr=0.005, final_lr=0.00001, mag_noise=1):
        super().__init__()
        self.initial_lr   = initial_lr
        self.final_lr     = final_lr
        self.total_epochs = total_epochs
        self.mag_noise    = mag_noise

    def on_epoch_begin(self, epoch, logs=None):
        decay_rate = self.final_lr / self.initial_lr
        decayed_lr = self.initial_lr * (decay_rate ** (epoch / self.total_epochs))

        noise = np.random.uniform(-1, 1) * self.mag_noise * np.sqrt(decayed_lr * self.initial_lr)
        noisy_lr = decayed_lr + noise

        if (noisy_lr < self.final_lr):
            noisy_lr = self.final_lr
        
        keras.backend.set_value(self.model.optimizer.lr, noisy_lr)
#====================================================================
class NoisySinLR(Callback):
    def __init__(self, total_epochs, init_lr=0.01, final_lr=0.00001, freq=10, mag_noise=1):
        super().__init__()
        """
        Custom learning rate scheduler that generates a sinusoidal learning rate.

        Args:
            init_lr (float): Initial learning rate.
            final_lr (float): Final learning rate.
            freq (float): Frequency of oscillation (in cycles per epoch).
            mag_noise (float): Magnitude of random noise to add to the sine wave.
        """
        self.total_epochs = total_epochs
        self.init_lr = init_lr
        self.final_lr = final_lr
        self.freq = freq
        self.mag_noise = mag_noise

    def on_epoch_begin(self, epoch, logs=None):
        line = self.init_lr + ((self.final_lr - self.init_lr) * epoch / self.total_epochs)
        sin  = np.cos((2 * np.pi / self.freq) * epoch) * self.init_lr - self.init_lr
        new_lr = line + self.mag_noise * np.random.uniform(size=1) * sin
        if (new_lr < self.final_lr):
            new_lr = self.final_lr
        keras.backend.set_value(self.model.optimizer.lr, new_lr)

# # Example usage:
# init_lr = 0.01
# final_lr = 0.0001
# frequency = 20  # Oscillate every 10 epochs
# mag_noise = 0.0005

       
# epochs = np.arange(0, 100)
# print(epochs)
# lrs    = np.ones(np.shape(epochs)[0], dtype=float) * -999
# #lrs = np.sin((2 * np.pi / frequency) * epochs)
# # noisy_lr = sine_lr + np.random.normal(scale=mag_noise)
# # lrs[i] = init_lr + (final_lr - init_lr) * noisy_lr
# #print(lrs)

# for i in range(np.shape(epochs)[0]):
#     line = init_lr + ((final_lr - init_lr) * epochs[i] / np.shape(epochs)[0])
#     sin  = np.cos((2 * np.pi / frequency) * epochs[i]) * init_lr - (init_lr)
#     lrs[i] = line + .1 * np.random.uniform(size=1) * sin

#     print(i, epochs[i], lrs[i])

# plt.scatter(epochs, lrs)
# plt.plot(epochs, init_lr + (final_lr - init_lr)/100*epochs, 'r-')
# plt.show()
 