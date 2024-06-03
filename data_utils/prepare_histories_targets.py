import numpy as np
import yaml
#====================================================================
def Histories_Targets(config_path, x_data, y_data, del_data=True):
    #----------------------------------------------------------------
    ''' Get variables from config '''

    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    history_len = int(config['HIST_TARG']['history_len'])
    target_len  = int(config['HIST_TARG']['target_len'])
    assert ((0 < history_len) and (history_len < np.shape(x_data)[0] - target_len)), "'history_len' must between between 1 and "+str(np.shape(x_data)[0] - target_len)+". Got: "+str(history_len)
    assert ((0 < target_len) and (target_len < np.shape(x_data)[0] - target_len)), "'target_len' must between between 1 and "+str(np.shape(x_data)[0] - history_len)+". Got: "+str(target_len)
    #----------------------------------------------------------------
    ''' Create the histories and targets '''

    histories = np.ones((np.shape(x_data)[0] - history_len - target_len, 
                         history_len, 
                         np.shape(x_data)[1],
                         np.shape(x_data)[2],
                         np.shape(x_data)[3]),
                         float) * -999
    
    targets   = np.ones((np.shape(y_data)[0] - history_len - target_len, 
                         target_len,
                         np.shape(y_data)[1],
                         np.shape(y_data)[2],
                         np.shape(y_data)[3]),
                         float) * -999

    for i in range(np.shape(histories)[0]):
        histories[i, ...] = x_data[i : i+history_len, ...]
        targets[i, ...]   = y_data[i+history_len : i+history_len+target_len, ...]

    #----------------------------------------------------------------
    if del_data:
        del(x_data)
        del(y_data)
    
    return histories, targets