import sys
sys.dont_write_bytecode = True
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import yaml

sys.path.append(os.getcwd())
from ml.train_test_keras import TrainKerasModel, TestKerasModel
from ml.train_test_naive_rf import TrainNaiveRF, TestNaiveRF
from ml.train_test_XGBoost import TrainNaiveXGBoost, TestNaiveXGBoost
#====================================================================
''' Get info from config '''
with open('config.yml', 'r') as c:
    config = yaml.load(c, Loader=yaml.FullLoader)

model_type      = config['MODEL_TYPE']
region          = config['REGION']
model_save_path = config['MODEL_SAVE_PATH']
model_pred_path = config['MODEL_PRED_PATH']

assert (region in ['Whole_Area', 'South_Land', 'North_Land', 'East_Ocean', 'West_Ocean']), "'region' must be 'Whole_Area', 'South_Land', 'North_Land', 'East_Ocean', 'West_Ocean'. Got: "+str(region)
assert (model_type in ['RF', 'GBM', 'Linear', 'Dense', 'Conv', 'LSTM', 'ConvLSTM', 'RBDN', 'Split', 'Denoise', 'DenoiseTrans', 'Trans']), "'model_type' must be 'RF', 'Dense', 'Conv', 'ConvLSTM', 'RBDN', 'Split', 'Denoise', 'DenoiseTrans', 'Trans'. Got: "+str(model_type)

short = {'Whole_Area':'WA', 'South_Land':'SL', 'North_Land':'NL', 'East_Ocean':'EO', 'West_Ocean':'WO'}
#====================================================================
if (model_type == 'Linear'):
    # TrainKerasModel('config.yml',
    #                 model_save_path=os.path.join(model_save_path, 'Linear'))
    TestKerasModel('config.yml',
                   model_name='Linear_reg=SL_f=1_In=OFTUVXYD_Out=O_e=10',
                   model_pred_path=model_pred_path)
#--------------------------------------------------------------------
elif (model_type == 'Dense'):
    e         = config['HYPERPARAMETERS']['dense_hyperparams_dict']['epochs']
    num_trans = config['HYPERPARAMETERS']['dense_hyperparams_dict']['num_trans']
    TrainKerasModel('config.yml',
                    prefix='Diamond',
                    model_save_path=os.path.join(model_save_path, 'Dense'))
    TestKerasModel('config.yml',
                   model_name='Diamond-Dense_reg='+short[region]+'_h=5_f=1_t='+str(num_trans)+'_In=OFTUVXYD_Out=O_e='+str(e),
                   model_pred_path=model_pred_path)
#--------------------------------------------------------------------   
elif (model_type == 'Conv'):
    e         = config['HYPERPARAMETERS']['conv_hyperparams_dict']['epochs']
    num_trans = config['HYPERPARAMETERS']['conv_hyperparams_dict']['num_trans']
    TrainKerasModel('config.yml',
                    model_save_path=os.path.join(model_save_path, 'Conv'))
    TestKerasModel('config.yml',
                   model_name='Conv_reg='+short[region]+'_h=5_f=1_t='+str(num_trans)+'_In=OTUVXYD_Out=O_e='+str(e),
                   model_pred_path=model_pred_path)
#--------------------------------------------------------------------
elif (model_type == 'LSTM'):
    e         = config['HYPERPARAMETERS']['lstm_hyperparams_dict']['epochs']
    num_trans = config['HYPERPARAMETERS']['lstm_hyperparams_dict']['num_trans']
    TrainKerasModel('config.yml',
                    model_save_path=os.path.join(model_save_path, 'LSTM'))
    TestKerasModel('config.yml',
                   model_name='LSTM_reg='+short[region]+'_h=5_f=1_t='+str(num_trans)+'_In=OFTUVXYD_Out=O_e='+str(e),
                   model_pred_path=model_pred_path)
#-------------------------------------------------------------------- 
elif (model_type == 'ConvLSTM'):
    e         = config['HYPERPARAMETERS']['convlstm_hyperparams_dict']['epochs']
    num_trans = config['HYPERPARAMETERS']['convlstm_hyperparams_dict']['num_trans']
    TrainKerasModel('config.yml',
                    model_save_path=os.path.join(model_save_path, 'ConvLSTM'))
    TestKerasModel('config.yml',
                   model_name='ConvLSTM_reg='+short[region]+'_h=5_f=1_t='+str(num_trans)+'_In=OFTUVXYD_Out=O_e='+str(e),
                   model_pred_path=model_pred_path)
#--------------------------------------------------------------------
elif (model_type == 'RBDN'):
    e = config['HYPERPARAMETERS']['rbdn_hyperparams_dict']['epochs']
    TrainKerasModel('config.yml',
                    model_save_path=os.path.join(model_save_path, 'RBDN'))
    TestKerasModel('config.yml',
                   model_name='RBDN_reg='+short[region]+'_h=5_f=1_In=OFTUVXYD_Out=O_e='+str(e),
                   model_pred_path=model_pred_path)
#--------------------------------------------------------------------
elif (model_type == 'Denoise'):
    print("DFKJBSDFB\nFIFSAPHOFSPDFPOUH\nAFSOFDSPOSFAPOIH\nAFOHFDOHIFSA")
    e = config['HYPERPARAMETERS']['denoise_hyperparams_dict']['epochs']
    TrainKerasModel('config.yml',
                    model_save_path=os.path.join(model_save_path, 'Denoise'))
    TestKerasModel('config.yml',
                   model_name='Denoise_reg='+short[region]+'_h=5_f=1_In=OFTUVXYD_Out=O_e='+str(e),
                   model_pred_path=model_pred_path)
#--------------------------------------------------------------------
elif (model_type == 'DenoiseTrans'):
    e         = config['HYPERPARAMETERS']['denoise_hyperparams_dict']['epochs']
    num_trans = config['HYPERPARAMETERS']['denoise_hyperparams_dict']['num_trans']
    TrainKerasModel('config.yml',
                    model_save_path=os.path.join(model_save_path, 'DenoiseTrans'))
    TestKerasModel('config.yml',
                   model_name='DenoiseTrans_reg='+short[region]+'_h=5_f=1_t='+str(num_trans)+'_In=OFTUVXYD_Out=O_e='+str(e),
                   model_pred_path=model_pred_path)
#-------------------------------------------------------------------- 
elif (model_type == 'Split'):
    e = config['HYPERPARAMETERS']['split_hyperparams_dict']['epochs']
    # TrainKerasModel('config.yml',
    #                 model_name='Split_reg='+short[region]+'_f=1_In=OFTUVXYD_Out=O_e='+str(e),
    #                 model_save_path=os.path.join(model_save_path, 'Split'))
    TestKerasModel('config.yml',
                   model_name='Split_reg='+short[region]+'_h=5_f=1_In=OFTUVXYD_Out=O_e='+str(e),
                   model_pred_path=model_pred_path)
#--------------------------------------------------------------------   
elif (model_type == 'Trans'):
    e = config['HYPERPARAMETERS']['trans_hyperssparams_dict']['epochs']
    TrainKerasModel('config.yml',
                    prefix='DiamondDense',
                    model_save_path=os.path.join(model_save_path, 'Trans'))
    TestKerasModel('config.yml',
                   model_name='DiamondDense-Trans_reg='+short[region]+'_h=5_f=1_In=OFTUVXYD_Out=O_e='+str(e),
                   model_pred_path=model_pred_path)
#--------------------------------------------------------------------
elif (model_type == 'RF'):
    # TrainNaiveRF('config.yml', 
    #              'data_utils/data_utils_config.yml', 
    #              model_save_path=os.path.join(model_save_path, 'RF'))
    TestNaiveRF('config.yml',
                'data_utils/data_utils_config.yml',
                model_name='XGBRF_reg='+short[region]+'_f=1_In=OFTUVXYD_Out=O',
                model_pred_path=model_pred_path)
#--------------------------------------------------------------------
elif (model_type == 'GBM'):
    # TrainNaiveXGBoost('config.yml', 
    #                   'data_utils/data_utils_config.yml', 
    #                   model_save_path=os.path.join(model_save_path, 'GBM'))
    TestNaiveXGBoost('config.yml',
                     'data_utils/data_utils_config.yml',
                     model_name='GBM_reg='+short[region]+'_f=1_In=OFTUVXYD_Out=O.pkl',
                     model_pred_path=model_pred_path)
#====================================================================
print("____________________________________________________________")
print(" >> >> Finished! << <<")
print("____________________________________________________________")
#====================================================================