import sys
sys.dont_write_bytecode = True
import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
from matplotlib.colors import Normalize, LogNorm, FuncNorm
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import fiona
import numpy as np
from datetime import datetime, timedelta
import yaml

sys.path.append(os.getcwd())
from ml.ml_utils import ParseModelName
#====================================================================
model_name = 'XGBRF_reg=WA_f=1_In=OFTUVXYD_Out=O'

date_idx = 0
#====================================================================
with open('config.yml', 'r') as c:
    config = yaml.load(c, Loader=yaml.FullLoader)

    model_pred_path = config['MODEL_PRED_PATH']
#====================================================================
''' Find y_pred '''
y_pred = 69
folders = os.listdir(model_pred_path)
if ('.DS_Store' in folders):
    folders.remove('.DS_Store')

for folder in folders:
    for root, dirs, files in os.walk(os.path.join(model_pred_path, folder)):
        for name in files:
            #print("root=",root, ", name=", name)
            if (model_name in name):
                if name.endswith('.npy'):
                    print(" >> Found:", os.path.join(root, name))
                    y_pred = np.load(os.path.join(root, name))

#====================================================================
''' Get info from model name '''
info, param_dict = ParseModelName(model_name)
short = {'Whole_Area':'WA', 'South_Land':'SL', 'North_Land':'NL', 'East_Ocean':'EO', 'West_Ocean':'WO'}
region = short[param_dict['REGION']]
model_type = param_dict['MODEL_TYPE']

if (param_dict['MODEL_TYPE'] == 'XGBRF'):
    model_type = 'RF'
#====================================================================
''' Find data '''
files = os.listdir(os.path.join(model_pred_path, 'Data'))
data  = 69
dates = 69
for file in files:
    if ((region in file) and (model_type in file)):
        if file.endswith('.npy'):
            data = np.load(os.path.join(os.path.join(model_pred_path, 'Data'), file))
            print(" >> Found:", os.path.join(os.path.join(model_pred_path, 'Data'), file))
    if ((model_type in file) and ('dates' in file)):
        if file.endswith('.csv'):
            dates = pd.read_csv(os.path.join(os.path.join(model_pred_path, 'Data'), file))
            print(" >> Found:", os.path.join(os.path.join(model_pred_path, 'Data'), file))

#====================================================================
try:
    world = gpd.read_file("/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")
except fiona.errors.DriverError:
    world = gpd.read_file("/content/drive/MyDrive/Colab_Notebooks/Data/ne_110m_land/ne_110m_land.shp")
    
lat = 69
lon = 69



#====================================================================





#====================================================================