import sys
sys.dont_write_bytecode = True
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import yaml

sys.path.append(os.getcwd())
from data_utils.data_loader import DataLoader
from data_utils.prepare_histories_targets import Histories_Targets
from data_utils.train_test_split import Train_Test_Split
#====================================================================
def NaiveRFDataLoader(config_path, data_config_path, return_shapes=False, shuffle=True):
    #----------------------------------------------------------------
    ''' Get data from config '''
    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    x_variables = config['HISTORY_DATA']
    for var in x_variables:
        assert (var in ['ozone', 'fire', 'temp', 'u-wind', 'v-wind', 'lon', 'lat', 'time']), var+" must be: 'ozone', 'fire', 'temp', 'u-wind', 'v-wind', 'lon', 'lat', 'time'"

    y_variables = config['TARGET_DATA']
    for var in y_variables:
        assert (var in ['ozone', 'fire', 'temp', 'u-wind', 'v-wind', 'lon', 'lat', 'time']), var+" must be: 'ozone', 'fire', 'temp', 'u-wind', 'v-wind', 'lon', 'lat', 'time'"

    offset = int(config['RF_OFFSET'])
    #----------------------------------------------------------------
    ''' Get training and target variables '''
    print(" >> Training data")
    x_data = DataLoader(config_path, data_config_path, 'HISTORY_DATA')
    print("____________________________________________________________")
    print(" >> Testing data")
    y_data = DataLoader(config_path, data_config_path, 'TARGET_DATA')

    print(" >> x_data", np.shape(x_data), ", y_data", np.shape(y_data))

    assert ((0 < offset) and (offset < np.shape(x_data)[0])), "'offset' must be between 1 and "+str(np.shape(x_data)[0])+". Got: "+str(offset)
    #----------------------------------------------------------------
    ''' Offset to put target data into the future'''

    x_data = x_data[:-offset, ...] # 1st dim is time
    y_data = y_data[offset:, ...]
    print(" >> x_data", np.shape(x_data), ", y_data", np.shape(y_data))
    #----------------------------------------------------------------
    ''' Train test split '''

    x_train, x_test, y_train, y_test = Train_Test_Split(config_path, 
                                                        x_data, 
                                                        y_data, 
                                                        shuffle=shuffle)

    x_train_orig_shape = np.shape(x_train)
    x_test_orig_shape  = np.shape(x_test)
    y_train_orig_shape = np.shape(y_train)
    y_test_orig_shape  = np.shape(y_test)
    #----------------------------------------------------------------
    ''' Transpose data '''

    x_train_T = np.transpose(x_train, (3, 0, 1, 2))
    x_train_flat = x_train_T.reshape(x_train_T.shape[0], -1)

    x_test_T = np.transpose(x_test, (3, 0, 1, 2))
    x_test_flat = x_test_T.reshape(x_test_T.shape[0], -1)

    y_train_T = np.transpose(y_train, (3, 0, 1, 2))
    y_train_flat = y_train_T.reshape(y_train_T.shape[0], -1)

    y_test_T = np.transpose(y_test, (3, 0, 1, 2))
    y_test_flat = y_test_T.reshape(y_test_T.shape[0], -1)

    print(np.shape(x_train_T), np.shape(x_train_flat), np.shape(x_test_T), np.shape(x_test_flat))
    print(np.shape(y_train_T), np.shape(y_train_flat), np.shape(y_test_T), np.shape(y_test_flat))
    #----------------------------------------------------------------
    ''' DataFrame '''

    x_train_df = pd.DataFrame(x_train_flat.T, columns=x_variables)
    y_train_df = pd.DataFrame(y_train_flat.T, columns=y_variables)

    x_test_df = pd.DataFrame(x_test_flat.T, columns=x_variables)
    y_test_df = pd.DataFrame(y_test_flat.T, columns=y_variables)

    del(x_data)
    del(y_data)
    del(x_train)
    del(y_train)
    del(x_train_T)
    del(x_train_flat)
    del(y_train_T)
    del(y_train_flat)
    del(x_test_T)
    del(x_test_flat)
    del(y_test_T)
    del(y_test_flat)

    if return_shapes:
        return x_train_df, x_test_df, y_train_df, y_test_df, x_train_orig_shape, x_test_orig_shape, y_train_orig_shape, y_test_orig_shape
    else:
        return x_train_df, x_test_df, y_train_df, y_test_df
#====================================================================
#x_train_df, x_test_df, y_train_df, y_test_df = NaiveRFDataLoader('config.yml')
#print(y_train_df.head())