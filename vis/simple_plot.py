import sys
sys.dont_write_bytecode = True
import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
from matplotlib.colors import Normalize, LogNorm, FuncNorm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import pandas as pd
import fiona
import numpy as np
from datetime import datetime, timedelta
import yaml

sys.path.append(os.getcwd())
from ml.ml_utils import ParseModelName
from misc.misc_utils import GetBoxCoords
from vis.plotting_utils import ShowYearMonth, truncate_datetime_v3, DegreeFormatter
#====================================================================
#model_name = 'Dense_reg=WA_h=5_f=1_t=1_In=OFTUVXYD_Out=O_e=10000.npy'
model_name = 'GBM_reg=WA_f=1_In=OFTUVXYD_Out=O'

date_idx = 222 # 222 for neural networks, 223 for RF
if (('GBM' in model_name) or ('RF' in model_name)):
    date_idx = 223

font = {'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)

text_fontsize = 16
big_font = 28
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
print("____________________________________________________________")
for folder in folders:
    for root, dirs, files in os.walk(os.path.join(model_pred_path, folder)):
        for name in files:
            #print("root=",root, ", name=", name)
            if (model_name in name):
                if name.endswith('.npy'):
                    print(" >> Found:", os.path.join(root, name))
                    y_pred = np.squeeze(np.load(os.path.join(root, name)))

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
''' Find data '''
files = os.listdir(os.path.join(model_pred_path, 'Data'))
data  = 69
dates = 69
for file in files:
    if ((region in file) and (model_type+'_' in file)):
        if file.endswith('.npy'):
            data = np.squeeze(np.load(os.path.join(os.path.join(model_pred_path, 'Data'), file)))
            print(" >> Found:", os.path.join(os.path.join(model_pred_path, 'Data'), file))
    if ((model_type+'_' in file) and ('dates' in file)):
        if file.endswith('.csv'):
            dates = pd.read_csv(os.path.join(os.path.join(model_pred_path, 'Data'), file))
            print(" >> Found:", os.path.join(os.path.join(model_pred_path, 'Data'), file))
#====================================================================

date_str = truncate_datetime_v3(dates.iloc[date_idx]['Days since 1970/1/1'])
print("____________________________________________________________")
print(" >> Date:", date_str)
#====================================================================
if ("_t=" in model_name):
    if (param_dict['num_trans']>0):
        model_type += '+Trans.'
#====================================================================
''' MSE '''
print("____________________________________________________________")
print(" >> MSE:", np.mean(np.square(np.subtract(data, y_pred))))
print("____________________________________________________________")
#====================================================================
''' Get lat/lon and world map '''

world = 69
try:
    world = gpd.read_file("/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")
except fiona.errors.DriverError:
    world = gpd.read_file("/content/drive/MyDrive/Colab_Notebooks/Data/ne_110m_land/ne_110m_land.shp")
    
boxes_dict = GetBoxCoords('data_utils/data_utils_config.yml')

min_lat = boxes_dict[param_dict['REGION']][3] + 0.25
max_lat = boxes_dict[param_dict['REGION']][1] - 0.25
min_lon = boxes_dict[param_dict['REGION']][0] + 0.5
max_lon = boxes_dict[param_dict['REGION']][2] - 0.5

lat = 69
lon = 69

lat = np.tile(np.arange(max_lat, min_lat-0.5, -0.5).reshape(-1, 1), (1, int(max_lon - min_lon+1)))
lon = np.tile(np.arange(min_lon, max_lon+1, 1).reshape(1, -1), (int((max_lat - min_lat)*2)+1, 1))

norm = Normalize(vmin=0, vmax=.036)

print(max(data.ravel()))
#====================================================================
''' Plot predictions '''

fig_map, ax_map = plt.subplots(1,1, dpi=100)
fig_map.set_size_inches(6, 6)

map = ax_map.scatter(x=lon, y=lat, c=y_pred[date_idx], s=150, cmap='bwr', norm=norm)

world.plot(ax=ax_map, facecolor='none', edgecolor='black', linewidth=.5, alpha=1)

# divider = make_axes_locatable(ax_map)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(map, cax=cax)

ax_map.set_xlim(min_lon-1, max_lon+1)
ax_map.set_ylim(min_lat-1, max_lat+1)

ax_map.set_xticks([0, 25, 50])
ax_map.set_yticks([20, 0, -20])

ax_map.xaxis.set_major_formatter(FuncFormatter(DegreeFormatter))
ax_map.yaxis.set_major_formatter(FuncFormatter(DegreeFormatter))

if (model_type in ['RF', 'XGBRF', 'GBM', 'Dense' ]):
    ax_map.set_title(model_type, fontsize=big_font, fontweight='bold')
    ax_map.set_xticklabels('')
elif (model_type in ['Conv', 'Conv+Trans.', 'LSTM', 'LSTM+Trans.', 'ConvLSTM']):
    ax_map.set_title(model_type, fontsize=big_font+2, fontweight='bold')
    ax_map.set_xticklabels('')
    ax_map.set_yticklabels('')
elif (model_type in ['Dense+Trans.', 'ConvLSTM+Trans.']):
    ax_map.set_title(model_type, fontsize=big_font-2, fontweight='bold')

fig_map.tight_layout()
#====================================================================
''' Plot data '''

fig_data, ax_data = plt.subplots(1,1, dpi=100)
fig_data.set_size_inches(6, 6)

map = ax_data.scatter(x=lon, y=lat, c=data[date_idx], s=150, cmap='bwr', norm=norm)

world.plot(ax=ax_data, facecolor='none', edgecolor='black', linewidth=.5, alpha=1)

ax_data.set_xlim(min_lon-1, max_lon+1)
ax_data.set_ylim(min_lat-1, max_lat+1)

ax_data.set_xticks([0, 25, 50])
ax_data.set_yticks([20, 0, -20])

ax_data.xaxis.set_major_formatter(FuncFormatter(DegreeFormatter))
ax_data.yaxis.set_major_formatter(FuncFormatter(DegreeFormatter))

ax_data.set_title("Data", fontsize=big_font, fontweight='bold')
ax_data.set_xticklabels('')

fig_data.tight_layout()
#====================================================================
''' Plot timeseries '''

# fig_time, ax_time = plt.subplots(1,1, dpi=100)
# fig_time.set_size_inches(10, 1)

# raw_ozone_mean = np.mean(data, axis=(1, 2))
# y_pred_mean    = np.mean(y_pred, axis=(1, 2))

# time_axis = 69

# try:
#     time_axis = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in dates['Days since 1970/1/1']]
# except ValueError:
#     time_axis = [datetime.strptime(date, '%Y-%m-%d') for date in dates['Days since 1970/1/1']]

# print(np.shape(time_axis), np.shape(raw_ozone_mean))

# ax_time.scatter(time_axis, raw_ozone_mean, s=15,color='black', marker='x', label='Data')
# ax_time.scatter(time_axis, y_pred_mean, s=20, facecolors='none', edgecolors='blue', marker='o', label='Prediction')

# ax_time.yaxis.set_label_position("right")
# ax_time.yaxis.tick_right()

# ShowYearMonth(ax_time, time_axis, start_line_idx=150, fontsize=22)
#====================================================================
# fig_bar, ax_bar = plt.subplots(1,1,dpi=100)
# fig_bar.set_size_inches(10, 10)

# font = {'weight' : 'bold',
#         'size'   : 11}

# plt.rc('font', **font)

# lol = ax_bar.scatter(x=1, y=0.015, c=0.015, norm=norm, cmap='bwr')

# divider = make_axes_locatable(ax_bar)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cbar = plt.colorbar(lol, cax=cax)
# cbar.set_label(r'$O_3$'+' '+r'$\left( \frac{mol}{m^2} \right)$', fontsize=20, rotation=270, labelpad=30)

#====================================================================
plt.show()
#====================================================================
if ("Trans." in model_type):
    model_type = model_type[:-1]

# fig_map.savefig(os.path.join('/Users/joshuamiller/Documents/Lancaster/Figs/SmallMaps', 
#                               'Pred_'+model_name+'_'+date_str+'.pdf'),
#                 bbox_inches='tight', pad_inches=0)

# fig_data.savefig(os.path.join('/Users/joshuamiller/Documents/Lancaster/Figs/SmallMaps', 
#                               'Data_'+model_type+'_'+date_str+'.pdf'),
#                 bbox_inches='tight', pad_inches=0)

# fig_time.savefig(os.path.join('/Users/joshuamiller/Documents/Lancaster/Figs', 
#                               'Time_'+model_name+'_'+date_str+'.pdf'),
#                 bbox_inches='tight', pad_inches=0)

# fig_bar.savefig(os.path.join('/Users/joshuamiller/Documents/Lancaster/Figs/SmallMaps', 
#                               'Colorbar.pdf'),
#                 bbox_inches='tight', pad_inches=0)