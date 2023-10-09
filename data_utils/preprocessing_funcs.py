import numpy as np
from sklearn import preprocessing

def Scale(data, reference, method='standard'):
    '''
    This function scales the data, either using the StandardScaler or MaxAbsScaler
    :param data: nparray, the data to be scaled 
    :param reference: nparray, what to use as a reference for the scaler, must be same size as data
    :param method: str, either 'standard' or 'maxabs', chooses the scaling method
    '''
    data = np.array(data)
    reference = np.array(reference)

    orig_shape = np.shape(data)

    if method == 'standard':
        scaler = preprocessing.StandardScaler() # Makes mean = 0, stdev = 1
    elif method == 'maxabs':
        scaler = preprocessing.MaxAbsScaler() # Scales to range [-1, 1], best for sparse data
    else:
        raise ValueError('Invalid method specified. Allowed values are "standard" and "maxabs".')

    scaler.fit(reference.ravel().reshape(-1, 1)) # Flatten array in order to obtain the mean over all the space and time

    scaled_data = scaler.transform(data.ravel().reshape(-1, 1))

    return scaled_data.reshape(orig_shape)