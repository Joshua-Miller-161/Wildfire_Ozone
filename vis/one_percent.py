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
from misc.misc_utils import GetBoxCoords, MADN
#====================================================================
model_name = 'ConvLSTM_reg=EO_h=5_f=1_t=0_In=OTUVXYD_Out=O_e=10000.npy'
#model_name = 'RF_reg=NL_f=1_In=OTUVXYD_Out=O'

date_idx = 222 # 222 for neural networks, 223 for RF
if (('GBM' in model_name) or ('RF' in model_name)):
    date_idx = 223

font = {'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)

text_fontsize = 16
big_font = 28
small_font = 23

cmap = 'PRGn'
#====================================================================
def create_binary_mask(arr, threshold=0.5):
    binary_mask = (arr > threshold).astype(int)
    return binary_mask
#====================================================================
def truncate_datetime_v3(dt_str):
    try:
        dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        dt = datetime.strptime(dt_str, '%Y-%m-%d')
    return dt.date().strftime('%Y-%m-%d')
#====================================================================
with open('config.yml', 'r') as c:
    config = yaml.load(c, Loader=yaml.FullLoader)

    model_pred_path = config['MODEL_PRED_PATH']
#====================================================================
''' Get info from model name '''
info, param_dict = ParseModelName(model_name)
short = {'Whole_Area':'WA', 'South_Land':'SL', 'North_Land':'NL', 'East_Ocean':'EO', 'West_Ocean':'WO'}
region = short[param_dict['REGION']]
model_type = param_dict['MODEL_TYPE']

if (param_dict['MODEL_TYPE'] == 'XGBRF'):
    model_type = 'RF'

if (param_dict['MODEL_TYPE'] == 'Diamond-Dense'):
    model_type = 'Dense'
#====================================================================
''' Find y_pred '''
y_pred = 69
folders = os.listdir(model_pred_path)
if ('.DS_Store' in folders):
    folders.remove('.DS_Store')
print("____________________________________________________________")
for folder in folders:
    for root, dirs, files in os.walk(os.path.join(model_pred_path, folder)):
        for name in files:
            #print("root=",root, ", name=", name)
            if (model_name in name and not ('v'+model_type+'_' in name)):
                if name.endswith('.npy'):
                    print(" >> Found:", os.path.join(root, name))
                    y_pred = np.squeeze(np.load(os.path.join(root, name)))
#====================================================================
''' Find data '''
files = os.listdir(os.path.join(model_pred_path, 'Data'))
data  = 69
dates = 69
for file in files:
    if ((region in file) and (''+model_type+'_' in file) and not ('v'+model_type+'_' in file)):
        if file.endswith('.npy'):
            data = np.squeeze(np.load(os.path.join(os.path.join(model_pred_path, 'Data'), file)))
            print(" >> Found:", os.path.join(os.path.join(model_pred_path, 'Data'), file))
    if ((model_type+'_' in file) and ('dates' in file) and not ('v'+model_type+'_' in file)):
        if file.endswith('.csv'):
            dates = pd.read_csv(os.path.join(os.path.join(model_pred_path, 'Data'), file))
            print(" >> Found:", os.path.join(os.path.join(model_pred_path, 'Data'), file))
#--------------------------------------------------------------------
# Get 5th percentile values
threshold = 0.02074 #np.percentile(data.ravel(), 99)

data_mask = create_binary_mask(data, threshold)

y_pred_mask = create_binary_mask(y_pred, threshold)

final_mask = np.multiply(data_mask, y_pred_mask)
#====================================================================
print("____________________________________________________________")
print(" >> Threshold =", round(threshold, 5), ", max(data) = ", round(max(data.ravel()), 5), ", sum(data) =", np.sum(data_mask), ", sum(y_pred) =", np.sum(y_pred_mask))
print("____________________________________________________________")
#====================================================================
if ("_t=" in model_name):
    if (param_dict['num_trans']>0):
        model_type += '+Trans'
#====================================================================
''' MSE '''
print("____________________________________________________________")
print(" >> MSE =", np.mean(np.square(np.subtract(data, y_pred))))
print("____________________________________________________________")
print(" >> Num. correct =", np.sum(final_mask), ", /day =", np.sum(final_mask) / np.shape(final_mask)[0])
print(" >> /area =", np.sum(final_mask) / (np.prod(np.shape(final_mask)[1:])), ", /(area * num. hotspots) =", np.sum(final_mask) / (np.prod(np.shape(final_mask)[1:]) * np.sum(data_mask)))
print(" >> % =", 100 * np.sum(final_mask) / np.sum(data_mask), ", sum(data) =", np.sum(data_mask))
print("____________________________________________________________")
print(" >>", model_name)
print("____________________________________________________________")
#====================================================================
