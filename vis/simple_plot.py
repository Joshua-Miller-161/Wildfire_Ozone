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
from misc.misc_utils import GetBoxCoords
from vis.plotting_utils import ShowYearMonth
#====================================================================
model_name = 'XGBRF_reg=SL_f=1_In=OFTUVXYD_Out=O'

date_idx = 77

font = {'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)
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

#====================================================================
''' Plot map '''
norm = Normalize(vmin=0, vmax=max(y_pred.ravel()))

fig_map, ax_map = plt.subplots(1,1, dpi=100)
fig_map.set_size_inches(6, 6)

map = ax_map.scatter(x=lon, y=lat, c=y_pred[date_idx], s=150, cmap='bwr', norm=norm)

world.plot(ax=ax_map, facecolor='none', edgecolor='black', linewidth=.5, alpha=1)

divider = make_axes_locatable(ax_map)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(map, cax=cax)

ax_map.set_xlim(min_lon-1, max_lon+1)
ax_map.set_ylim(min_lat-1, max_lat+1)
#ax_map.set_title(dates.iloc[date_idx]['Days since 1970/1/1'])
ax_map.yaxis.set_label_position("right")
ax_map.yaxis.tick_right()

fig_map.tight_layout()
#====================================================================
''' Plot timeseries '''

fig_time, ax_time = plt.subplots(1,1, dpi=100)
fig_time.set_size_inches(10, 1)

raw_ozone_mean = np.mean(data, axis=(1, 2))
y_pred_mean    = np.mean(y_pred, axis=(1, 2))

time_axis = [datetime.strptime(date, '%Y-%m-%d') for date in dates['Days since 1970/1/1']]

ax_time.scatter(time_axis, raw_ozone_mean, s=15,color='black', marker='x', label='Data')
ax_time.scatter(time_axis, y_pred_mean, s=20, facecolors='none', edgecolors='blue', marker='o', label='Prediction')

ax_time.yaxis.set_label_position("right")
ax_time.yaxis.tick_right()

ShowYearMonth(ax_time, time_axis, start_line_idx=150, fontsize=22)
#====================================================================
plt.show()

fig_map.savefig(os.path.join('/Users/joshuamiller/Documents/Lancaster/Figs', 
                              'map_'+model_name+'_'+dates.iloc[date_idx]['Days since 1970/1/1']+'.pdf'),
                bbox_inches='tight', pad_inches=0)

fig_time.savefig(os.path.join('/Users/joshuamiller/Documents/Lancaster/Figs', 
                              'time_'+model_name+'_'+dates.iloc[date_idx]['Days since 1970/1/1']+'.pdf'),
                bbox_inches='tight', pad_inches=0)