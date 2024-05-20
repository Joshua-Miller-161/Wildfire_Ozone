import sys
sys.dont_write_bytecode = True
import numpy as np
from sklearn import preprocessing
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
import json
import os
#====================================================================
def Scale(data, method='standard', scale_factor=1, lower=0, upper=6, del_data=True, to_file=False, to_file_path='data_utils/scale_files', data_name=None):
    '''
    This function scales the data, either using the StandardScaler or MaxAbsScaler
    :param data: nparray, the data to be scaled 
    :param reference: nparray, what to use as a reference for the scaler, must be same size as data
    :param method: str, either 'standard', 'maxabs', 'minmax', or 'log', chooses the scaling method
    :param to_file: Bool, whether to write key aspects of data to .json file to allow for inversion to original
    :to_file_path: str, path to save .json files to
    :param data_name: str, the name of the dataset
    '''

    data = np.array(data)
    print(" - Raw data min:", round(min(data.ravel()), 8), ", max:", round(max(data.ravel()), 8), ", avg.:", round(np.mean(data.ravel()), 8), ", stdev:", round(np.std(data.ravel()), 8))

    orig_shape = np.shape(data)

    scaled_data = -999
    #----------------------------------------------------------------
    if (method == 'standard'):
        scaler = preprocessing.StandardScaler() # Makes mean = 0, stdev = 1
        scaler.fit(data.ravel().reshape(-1, 1)) # Flatten array in order to obtain the mean over all the space and time
        scaled_data = scaler.transform(data.ravel().reshape(-1, 1))

        dict_ = {'orig_mean': np.mean(data.ravel()), 'orig_stdev': np.std(data.ravel())}
        with open(os.path.join(to_file_path, data_name+'_'+method+'.json'), 'w') as f:
            json.dump(dict_, f)
    #----------------------------------------------------------------
    elif (method == 'maxabs'):
        scaler = preprocessing.MaxAbsScaler() # Scales to range [-1, 1], best for sparse data
        scaler.fit(data.ravel().reshape(-1, 1)) # Flatten array in order to obtain the mean over all the space and time
        scaled_data = scaler.transform(data.ravel().reshape(-1, 1)) * scale_factor

        dict_ = {'orig_max': np.max(data.ravel())}
        with open(os.path.join(to_file_path, data_name+'_'+method+'.json'), 'w') as f:
            json.dump(dict_, f)
    #----------------------------------------------------------------
    elif (method == 'minmax'):
        data_min = min(data.ravel())
        data_max = max(data.ravel())
        scaled_data = (data - data_min) / (data_max - data_min) * (upper - lower) + lower

        dict_ = {'orig_min': np.min(data.ravel()), 
                 'orig_max': np.max(data.ravel()),
                 'upper': upper,
                 'lower': lower}
        with open(os.path.join(to_file_path, data_name+'_'+method+'.json'), 'w') as f:
            json.dump(dict_, f)
    #----------------------------------------------------------------
    elif (method == 'log'):
        scaled_data = data.ravel()
        min_ = 999
        for i in range(np.shape(scaled_data)[0]):
            if (scaled_data[i] > 0):
                if (np.log10(scaled_data[i]) < min_):
                    min_ = np.log10(scaled_data[i])

        for j in range(np.shape(scaled_data)[0]):
            if (scaled_data[j] > 0):
                scaled_data[j] = np.log10(scaled_data[j]) * scale_factor

        dict_ = {'scale_factor': scale_factor}
        with open(os.path.join(to_file_path, data_name+'_'+method+'.json'), 'w') as f:
            json.dump(dict_, f)
    #----------------------------------------------------------------
    else:
        raise ValueError('Invalid method specified. Allowed values are "standard" and "maxabs".')

    if del_data:
        del(data)
    
    print(" - Scaled data min:", round(min(scaled_data.ravel()), 8), ", max:", round(max(scaled_data.ravel()), 8), ", avg.:", round(np.mean(scaled_data.ravel()), 8), ", stdev:", round(np.std(scaled_data.ravel()), 8))
    return scaled_data.reshape(orig_shape)
#====================================================================
def UnScale(scaled_data, json_file):
    orig_shape = np.shape(scaled_data)

    scaled_data_flat = np.asarray(scaled_data.ravel())
    del(scaled_data)
    data = -999
    #----------------------------------------------------------------
    if ('standard' in json_file):
        dict_ = json.load(open(json_file))
        data  = dict_['orig_stdev'] * scaled_data_flat + dict_['orig_mean']
    #----------------------------------------------------------------
    elif ('maxabs' in json_file):
        dict_ = json.load(open(json_file))
        data  = dict_['orig_max'] * scaled_data_flat
    #----------------------------------------------------------------
    elif ('minmax' in json_file):
        
        dict_  = json.load(open(json_file))
        data   = (scaled_data_flat + dict_['lower']) * (dict_['orig_max'] - dict_['orig_min']) / (dict_['upper'] - dict_['lower']) + dict_['orig_min']
    #----------------------------------------------------------------
    elif ('log' in json_file):
        dict_ = json.load(open(json_file))
        data  = np.zeros_like(scaled_data_flat)
        for i in range(np.shape(scaled_data_flat)[0]):
            if not (scaled_data_flat[i] == 0):
                data[i] = 10**(scaled_data_flat[i] / dict_['scale_factor'])
    #----------------------------------------------------------------
    return data.reshape(orig_shape)
#====================================================================
def DoKrig(x, y, val, x_target, y_target):
    OK = OrdinaryKriging(
                x,
                y,
                val,
                variogram_model='spherical',
                verbose=False,
                enable_plotting=False,
                nlags=10)
    return OK.execute("points", x_target, y_target)
#====================================================================
def DownSample(data, downsample_rate, axis, delete=False):
    '''
    Made by ChatGPT - Extracts data points separated by skip along the given axis

    Returns downsampled data.
    
    - data (ndarray) - the data
    - skip (int) - the number of elements that are skiped when downsampling
    - axis (int) - the axis on which to downsample
    - delete (bool, optional) - whether or not to delete the original data in order to save memory
    '''
    slices       = [slice(None)] * data.ndim
    slices[axis] = slice(None, None, downsample_rate)
    new_data     = data[tuple(slices)]
    
    print('Orig. shape :', np.shape(data), "----> new shape :", np.shape(new_data))

    if delete:
        del(data)

    return new_data