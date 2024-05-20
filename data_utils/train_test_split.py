import numpy as np
import yaml
#====================================================================
def Train_Test_Split(config_path, x_data, y_data, shuffle=True, del_data=True):
    #----------------------------------------------------------------
    ''' Get variables from config '''

    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    split = float(config['TRAIN_TEST'])
    assert ((0 < split) and (split < 1)), "'TRAIN_TEST' must between between 0 and 1. Got: "+str(split)

    assert (np.shape(x_data)[0] == np.shape(y_data)[0]), "np.shape(x_data)[0] must equal np.shape(y_data)[0]. Got: "+str(np.shape(x_data)[0])+", "+str(np.shape(y_data)[0])
    #----------------------------------------------------------------
    ''' Split data '''

    split_idx = int(split * np.shape(x_data)[0])

    if shuffle:
        permutation = np.random.permutation(x_data.shape[0])
        x_data = x_data[permutation]
        y_data = y_data[permutation]

    x_train = x_data[:split_idx, ...]
    x_test  = x_data[split_idx:, ...]
    y_train = y_data[:split_idx, ...]
    y_test  = y_data[split_idx:, ...]

    if del_data:
        del(x_data)
        del(y_data)
    
    return x_train, x_test, y_train, y_test
#====================================================================