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
model_name = 'ConvLSTM_reg=NL_h=5_f=1_t=1_In=OFTUVXYD_Out=O_e=10000.npy'
#model_name = 'XGBRF_reg=NL_f=1_In=OFTUVXYD_Out=O'

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

date_str = truncate_datetime_v3(dates.iloc[date_idx]['Days since 1970/1/1'])
print("____________________________________________________________")
print(" >> Date:", date_str)
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
print("____________________________________________________________")
print(" >>", model_name)
print("____________________________________________________________")
#====================================================================
# ''' Get lat/lon and world map '''

# world = 69
# try:
#     world = gpd.read_file("/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")
# except fiona.errors.DriverError:
#     world = gpd.read_file("/content/drive/MyDrive/Colab_Notebooks/Data/ne_110m_land/ne_110m_land.shp")
    
# boxes_dict = GetBoxCoords('data_utils/data_utils_config.yml')

# min_lat = boxes_dict[param_dict['REGION']][3] + 0.25
# max_lat = boxes_dict[param_dict['REGION']][1] - 0.25
# min_lon = boxes_dict[param_dict['REGION']][0] + 0.5
# max_lon = boxes_dict[param_dict['REGION']][2] - 0.5

# lat = 69
# lon = 69

# lat = np.tile(np.arange(max_lat, min_lat-0.5, -0.5).reshape(-1, 1), (1, int(max_lon - min_lon+1)))
# lon = np.tile(np.arange(min_lon, max_lon+1, 1).reshape(1, -1), (int((max_lat - min_lat)*2)+1, 1))

# max_ = max([max(y_pred_madn.ravel()), max(data_madn.ravel())])
# norm = Normalize(vmin=0, vmax=1 * max_)

# max_diff = max(abs(data_madn.ravel() - y_pred_madn.ravel()))
# norm_diff = Normalize(vmin=-max_diff, vmax=max_diff)
# #====================================================================
# ''' Plot predictions '''

# fig_map, ax_map = plt.subplots(1,1, dpi=100)
# fig_map.set_size_inches(6, 6)

# map = ax_map.scatter(x=lon, y=lat, c=y_pred_madn[date_idx], s=150, cmap='Reds', norm=norm)

# world.plot(ax=ax_map, facecolor='none', edgecolor='black', linewidth=.5, alpha=1)

# divider = make_axes_locatable(ax_map)
# cax = divider.append_axes("right", size="5%", pad=0.15)
# cbar = plt.colorbar(map, cax=cax)
# #cbar.set_label('MADN')

# ax_map.set_xlim(min_lon-1, max_lon+1)
# ax_map.set_ylim(min_lat-1, max_lat+1)

# if (model_type in ['RF', 'XGBRF', 'GBM', 'Dense']):
#     ax_map.set_title(model_type, fontsize=big_font, fontweight='bold')
#     ax_map.set_xticklabels('')
# elif (model_type in ['Conv', 'Conv+Trans', 'LSTM', 'LSTM+Trans', 'ConvLSTM']):
#     ax_map.set_title(model_type, fontsize=big_font-6, fontweight='bold')
#     ax_map.set_xticklabels('')
#     ax_map.set_yticklabels('')
# elif (model_type in ['Dense+Trans', 'ConvLSTM+Trans']):
#     ax_map.set_title(model_type, fontsize=small_font, fontweight='bold')

# ax_map.text(.95 * min_lon, .85 * max_lat, "Avg. MADN = "+str(round(np.mean(y_pred_madn[date_idx]) , 7)), 
#              fontsize=text_fontsize-2)

# fig_map.tight_layout()
# #====================================================================
# ''' Plot data '''

# fig_data, ax_data = plt.subplots(1,1, dpi=100)
# fig_data.set_size_inches(6, 6)

# map = ax_data.scatter(x=lon, y=lat, c=data_madn[date_idx], s=150, cmap='Reds', norm=norm)

# world.plot(ax=ax_data, facecolor='none', edgecolor='black', linewidth=.5, alpha=1)


# ax_data.set_xlim(min_lon-1, max_lon+1)
# ax_data.set_ylim(min_lat-1, max_lat+1)
# ax_data.set_title("Data", fontsize=big_font, fontweight='bold')
# ax_data.set_xticklabels('')
# ax_data.text(.95 * min_lon, .85 * max_lat, "Avg. MADN = "+str(round(np.mean(data_madn[date_idx]) , 7)), 
#              fontsize=text_fontsize)

# fig_data.tight_layout()

#====================================================================
plt.show()
#====================================================================
# fig_map.savefig(os.path.join('/Users/joshuamiller/Documents/Lancaster/Figs/MADNMaps', 
#                               'MADN_Pred_'+model_name+'_'+date_str+'.pdf'),
#                 bbox_inches='tight', pad_inches=0)

# fig_data.savefig(os.path.join('/Users/joshuamiller/Documents/Lancaster/Figs/MADNMaps', 
#                               'MADN_Data_'+model_type+'_'+date_str+'.pdf'),
#                 bbox_inches='tight', pad_inches=0)
