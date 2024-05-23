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
from ml.train_test_keras import TrainConvLSTM, TestConvLSTM, TrainLinear
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
if (model_type == 'ConvLSTM'):
    TrainConvLSTM('config.yml',
                  model_save_path=os.path.join(model_save_path, 'ConvLSTM'))
    #TestConvLSTM('config.yml', 'ConvLSTM')

elif (model_type == 'Linear'):
    TrainLinear('config.yml',
                model_save_path=os.path.join(model_save_path, 'Linear'))

elif (model_type == 'RF'):
    TrainNaiveRF('config.yml', 
                 'data_utils/data_utils_config.yml', 
                 model_save_path=os.path.join(model_save_path, 'RF'))
#====================================================================
# ''' Prepare data '''

print("AHHHHHHHEFJENF:AEONFWENFWE:OKFNWE\nDIE:OFNW:OEFNWE:OFJNEWOFNWE")
#====================================================================



