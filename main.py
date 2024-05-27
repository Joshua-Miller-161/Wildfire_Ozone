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
#====================================================================
''' Get infor from config '''
with open('config.yml', 'r') as c:
    config = yaml.load(c, Loader=yaml.FullLoader)

model_type = config['MODEL_TYPE']
region = config['REGION']
model_save_path = config['MODEL_SAVE_PATH']

assert (region in ['Whole_Area', 'South_Land', 'North_Land', 'East_Ocean', 'West_Ocean']), "'region' must be 'Whole_Area', 'South_Land', 'North_Land', 'East_Ocean', 'West_Ocean'. Got: "+str(region)
assert (model_type in ['RF', 'Linear', 'Dense', 'ConvLSTM', 'Trans']), "'model_type' must be 'RF', 'Dense', 'ConvLSTM', 'Trans'. Got: "+str(model_type)
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
                    model_name='DiamondDense_reg=SL_f=1_In=OFTUVXYD_Out=O_e='+str(e),
                    model_save_path=os.path.join(model_save_path, 'Dense'))
    TestKerasModel('config.yml',
                   model_name='DiamondDense_reg=SL_f=1_In=OFTUVXYD_Out=O_e='+str(e))
#--------------------------------------------------------------------   
elif (model_type == 'ConvLSTM'):
    e = config['HYPERPARAMETERS']['trans_hyperparams_dict']['epochs']
    TrainKerasModel('config.yml',
                    model_save_path=os.path.join(model_save_path, 'ConvLSTM'))
    TestKerasModel('config.yml',
                   model_name='ConvLSTM_reg=NL_f=1_In=OFTUVXYD_Out=O_e=30')
#--------------------------------------------------------------------   
elif (model_type == 'Trans'):
    e = config['HYPERPARAMETERS']['trans_hyperparams_dict']['epochs']
    TrainKerasModel('config.yml',
                    model_name='DiamondDenseTrans_reg=WO_f=1_In=OFTUVXYD_Out=O_e='+str(e),
                    model_save_path=os.path.join(model_save_path, 'Trans'))
    TestKerasModel('config.yml',
                   model_name='DiamondDenseTrans_reg=WO_f=1_In=OFTUVXYD_Out=O_e='+str(e))
#--------------------------------------------------------------------
elif (model_type == 'RF'):
    TrainNaiveRF('config.yml', 
                 'data_utils/data_utils_config.yml', 
                 model_save_path=os.path.join(model_save_path, 'RF'))
#====================================================================
# ''' Prepare data '''
print("____________________________________________________________")
print(" >> >> Finished! << <<")
print("____________________________________________________________")
#====================================================================