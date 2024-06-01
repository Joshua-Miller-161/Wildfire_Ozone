import sys
sys.dont_write_bytecode = True
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import yaml

sys.path.append(os.getcwd())
from data_utils.data_loader import DataLoader
from data_utils.prepare_histories_targets import Histories_Targets
from data_utils.train_test_split import Train_Test_Split
from ml.train_test_keras import TrainKerasModel, TestKerasModel
from ml.train_test_naive_rf import TrainNaiveRF, TestNaiveRF
from ml.train_test_XGBoost import TrainNaiveXGBoost, TestNaiveXGBoost
#====================================================================
''' Get infor from config '''
with open('config.yml', 'r') as c:
    config = yaml.load(c, Loader=yaml.FullLoader)

model_type = config['MODEL_TYPE']
region = config['REGION']
model_save_path = config['MODEL_SAVE_PATH']

assert (region in ['Whole_Area', 'South_Land', 'North_Land', 'East_Ocean', 'West_Ocean']), "'region' must be 'Whole_Area', 'South_Land', 'North_Land', 'East_Ocean', 'West_Ocean'. Got: "+str(region)
assert (model_type in ['RF', 'GBM', 'Linear', 'Dense', 'ConvLSTM', 'Trans']), "'model_type' must be 'RF', 'Dense', 'ConvLSTM', 'Trans'. Got: "+str(model_type)


short = {'Whole_Area':'WO', 'South_Land':'SL', 'North_Land':'NL', 'East_Ocean':'EO', 'West_Ocean':'WO'}

#====================================================================
if (model_type == 'Linear'):
    # TrainKerasModel('config.yml',
    #                 model_save_path=os.path.join(model_save_path, 'Linear'))
    TestKerasModel('config.yml',
                   model_name='Linear_reg=SL_f=1_In=OFTUVXYD_Out=O_e=10')
#--------------------------------------------------------------------
elif (model_type == 'Dense'):
    e = config['HYPERPARAMETERS']['trans_hyperparams_dict']['epochs']
    TrainKerasModel('config.yml',
                    prefix='Diamond',
                    model_save_path=os.path.join(model_save_path, 'Dense'))
    TestKerasModel('config.yml',
                   model_name='Dense_reg='+short[region]+'_f=1_In=OFTUVXYD_Out=O_e=100',
                   model_folder=model_save_path)
#--------------------------------------------------------------------   
elif (model_type == 'ConvLSTM'):
    e = config['HYPERPARAMETERS']['trans_hyperparams_dict']['epochs']
    TrainKerasModel('config.yml',
                    model_save_path=os.path.join(model_save_path, 'ConvLSTM'))
    TestKerasModel('config.yml',
                   model_name='ConvLSTM_reg='+short[region]+'_f=1_In=OFTUVXYD_Out=O_e='+str(e),
                   model_folder=model_save_path)
#--------------------------------------------------------------------   
elif (model_type == 'Trans'):
    e = config['HYPERPARAMETERS']['trans_hyperparams_dict']['epochs']
    TrainKerasModel('config.yml',
                    model_name='DiamondDenseTrans_reg='+short[region]+'_f=1_In=OFTUVXYD_Out=O_e='+str(e),
                    model_save_path=os.path.join(model_save_path, 'Trans'))
    TestKerasModel('config.yml',
                   model_name='DiamondDenseTrans_reg='+short[region]+'_f=1_In=OFTUVXYD_Out=O_e='+str(e))
#--------------------------------------------------------------------
elif (model_type == 'RF'):
    TrainNaiveRF('config.yml', 
                 'data_utils/data_utils_config.yml', 
                 model_save_path=os.path.join(model_save_path, 'RF'))
    TestNaiveRF('config.yml', 
                 'data_utils/data_utils_config.yml', 
                 model_name='XGBRF_reg=SL_f=1_In=OFTUVXYD_Out=O.pkl')
#--------------------------------------------------------------------
elif (model_type == 'GBM'):
    TrainNaiveXGBoost('config.yml', 
                      'data_utils/data_utils_config.yml', 
                      model_save_path=os.path.join(model_save_path, 'GBM'))
    TestNaiveXGBoost('config.yml', 
                      'data_utils/data_utils_config.yml',
                      model_name='GBM_reg=SL_f=1_In=OFTUVXYD_Out=O.pkl')
#====================================================================
# ''' Prepare data '''
print("____________________________________________________________")
print(" >> >> Finished! << <<")
print("____________________________________________________________")
#====================================================================