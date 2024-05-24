import sys
sys.dont_write_bytecode = True
import numpy as np
import os
import yaml
import pandas as pd
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
    epochs       = config['HYPERPARAMETERS']['dense_hyperparams_dict']['epochs']
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
        print("model_name:", model_name)

        for var in history_vars:
            model_name += shorthand_dict[var]

        print("model_name:", model_name)

        model_name += '_Out='
        for var in target_vars:
            model_name += shorthand_dict[var]
        
        model_name += '.joblib'
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif (config['MODEL_TYPE'] == 'Linear'):
        model_name = 'Linear_reg='+shorthand_dict[region]+'_f='+str(int(rf_offset))+'_In='
        print("model_name:", model_name)

        for var in history_vars:
            model_name += shorthand_dict[var]

        model_name += '_Out='
        for var in target_vars:
            model_name += shorthand_dict[var]
        
        model_name += '_e='+str(epochs)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif (config['MODEL_TYPE'] == 'Dense'):
        model_name = 'Dense_reg='+shorthand_dict[region]+'_f='+str(int(rf_offset))+'_In='
        print("model_name:", model_name)

        for var in history_vars:
            model_name += shorthand_dict[var]

        model_name += '_Out='
        for var in target_vars:
            model_name += shorthand_dict[var]
        
        model_name += '_e='+str(epochs)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif (config['MODEL_TYPE'] == 'ConvLSTM'):
        model_name = 'CONVLSTM_reg='+shorthand_dict[region]+'_f='+str(int(rf_offset))+'_In='
        print("model_name:", model_name)

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