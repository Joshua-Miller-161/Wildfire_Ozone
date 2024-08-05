import sys
sys.dont_write_bytecode = True
import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize, LogNorm, FuncNorm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import pandas as pd
import geopandas as gpd
import fiona
import numpy as np
from datetime import datetime, timedelta
import yaml

sys.path.append(os.getcwd())
from ml.ml_utils import ParseModelName
from misc.misc_utils import GetBoxCoords, MADN
from vis.plotting_utils import truncate_datetime_v3, DegreeFormatter
#====================================================================
model_name = 'ConvLSTM_reg=WA_h=5_f=1_t=1_In=OTUVXYD_Out=O_e=10000.npy'
#model_name = 'GBM_reg=WO_f=1_In=OTUVXYD_Out=O'

date_idx = 222 # 222 for neural networks, 223 for RF
if (('GBM' in model_name) or ('RF' in model_name)):
    date_idx = 223

font = {'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)

text_fontsize = 16
big_font = 28
small_font = 23

cmap = LinearSegmentedColormap.from_list('custom', ['white', 'darkorange', 'orangered', 'red', 'darkred'], N=12)
cmap_diff = 'PRGn'
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
#--------------------------------------------------------------------
# Get MADN values
y_pred_madn = np.ones_like(y_pred) * -999

for i in range(y_pred_madn.shape[0]):
    y_pred_madn[i] = MADN(y_pred[i])

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
# Get MADN values
data_madn = np.ones_like(data) * -999

for i in range(data_madn.shape[0]):
    data_madn[i] = MADN(data[i])
#====================================================================
print("____________________________________________________________")
print(" >> data MADN:", np.mean(data_madn), ", pred. MADN:", np.mean(y_pred_madn))
print("____________________________________________________________")
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
print(" >> MSE(MADN):", np.mean(np.square(np.subtract(data_madn, y_pred_madn))))
print("____________________________________________________________")
print(" >>", model_name)
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

max_ = .012 #max([max(y_pred_madn.ravel()), max(data_madn.ravel())])
norm = Normalize(vmin=0, vmax=1 * max_)

max_diff = max(abs(data_madn.ravel() - y_pred_madn.ravel()))
norm_diff = Normalize(vmin=-max_diff, vmax=max_diff)
#====================================================================
''' Plot predictions '''

fig_map, ax_map = plt.subplots(1,1, dpi=100)
fig_map.set_size_inches(6, 6)

map = ax_map.scatter(x=lon, y=lat, c=y_pred_madn[date_idx], s=150, cmap=cmap, norm=norm)

world.plot(ax=ax_map, facecolor='none', edgecolor='black', linewidth=.5, alpha=1)

# divider = make_axes_locatable(ax_map)
# cax = divider.append_axes("right", size="5%", pad=0.15)
# cbar = plt.colorbar(map, cax=cax)
#cbar.set_label('MADN')

ax_map.set_xlim(min_lon-1, max_lon+1)
ax_map.set_ylim(min_lat-1, max_lat+1)

ax_map.set_xticks([0, 25, 50])
ax_map.set_yticks([20, 0, -20])

ax_map.xaxis.set_major_formatter(FuncFormatter(DegreeFormatter))
ax_map.yaxis.set_major_formatter(FuncFormatter(DegreeFormatter))

if (model_type in ['RF', 'XGBRF', 'GBM', 'Dense' ]):
    ax_map.set_title(model_type, fontsize=big_font, fontweight='bold')
    ax_map.text(.95 * min_lon, .85 * max_lat, "Avg. MADN = "+str(round(np.mean(y_pred_madn[date_idx]) , 7)), 
                fontsize=text_fontsize)
    ax_map.set_xticklabels('')
elif (model_type in ['Conv', 'Conv+Trans.', 'LSTM', 'LSTM+Trans.', 'ConvLSTM']):
    ax_map.set_title(model_type, fontsize=big_font+1, fontweight='bold')
    ax_map.text(.95 * min_lon, .85 * max_lat, "Avg. MADN = "+str(round(np.mean(y_pred_madn[date_idx]) , 7)), 
                fontsize=text_fontsize+1.5)
    ax_map.set_xticklabels('')
    ax_map.set_yticklabels('')
elif (model_type in ['Dense+Trans.', 'ConvLSTM+Trans.']):
    ax_map.set_title(model_type, fontsize=big_font-2, fontweight='bold')
    ax_map.text(.95 * min_lon, .85 * max_lat, "Avg. MADN = "+str(round(np.mean(y_pred_madn[date_idx]) , 7)), 
                fontsize=text_fontsize-1)

fig_map.tight_layout()
#====================================================================
''' Plot data '''

fig_data, ax_data = plt.subplots(1,1, dpi=100)
fig_data.set_size_inches(6, 6)

map = ax_data.scatter(x=lon, y=lat, c=data_madn[date_idx], s=150, cmap=cmap, norm=norm)

world.plot(ax=ax_data, facecolor='none', edgecolor='black', linewidth=.5, alpha=1)


ax_data.set_xlim(min_lon-1, max_lon+1)
ax_data.set_ylim(min_lat-1, max_lat+1)

ax_data.set_xticks([0, 25, 50])
ax_data.set_yticks([20, 0, -20])

ax_data.xaxis.set_major_formatter(FuncFormatter(DegreeFormatter))
ax_data.yaxis.set_major_formatter(FuncFormatter(DegreeFormatter))

ax_data.set_title("Data", fontsize=big_font, fontweight='bold')
ax_data.set_xticklabels('')
ax_data.text(.95 * min_lon, .85 * max_lat, "Avg. MADN = "+str(round(np.mean(data_madn[date_idx]) , 7)), 
             fontsize=text_fontsize)

fig_data.tight_layout()

#====================================================================
# ''' Plot difference '''
# fig_diff, ax_diff = plt.subplots(1,1, dpi=100)
# fig_diff.set_size_inches(6, 6)

# diff = ax_diff.scatter(x=lon, y=lat, c=data_madn[date_idx] - y_pred_madn[date_idx], s=150, cmap=cmap_diff, norm=norm_diff)

# world.plot(ax=ax_diff, facecolor='none', edgecolor='black', linewidth=.5, alpha=1)

# ax_diff.set_xlim(min_lon-1, max_lon+1)
# ax_diff.set_ylim(min_lat-1, max_lat+1)
# ax_diff.set_xticks([0, 25, 50])
# ax_diff.set_yticks([20, 0, -20])

# if (model_type in ['RF', 'XGBRF', 'GBM', 'Dense', 'Dense+Trans']):
#     ax_diff.set_title('Data - '+model_type, fontsize=big_font-2, fontweight='bold')
#     ax_diff.set_xticklabels('')
# elif (model_type in ['LSTM', 'LSTM+Trans', 'ConvLSTM', 'ConvLSTM+Trans']):
#     ax_diff.set_title('Data - '+model_type, fontsize=big_font+1, fontweight='bold')
#     ax_diff.set_xticklabels('')
#     ax_diff.set_yticklabels('')
# elif (model_type in ['Conv']):
#     ax_diff.set_title('Data - '+model_type, fontsize=small_font, fontweight='bold')
# elif (model_type in ['Conv+Trans']):
#     ax_diff.set_title('Data - '+model_type, fontsize=big_font, fontweight='bold')
#     ax_diff.set_yticklabels('')
# # ax_diff.text(.95 * min_lon, .85 * max_lat, "Avg. = "+str(round(np.mean(y_pred_madn[date_idx]) , 7)), 
# #              fontsize=text_fontsize)

# fig_diff.tight_layout()

#====================================================================
fig_bar, ax_bar = plt.subplots(1, 1, dpi=100)
fig_bar.set_size_inches(10, 10)

font = {'weight' : 'bold',
        'size'   : 12}

plt.rc('font', **font)

lol = ax_bar.scatter(x=1, y=0.015, c=0.015, norm=norm, cmap=cmap)

divider = make_axes_locatable(ax_bar)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(lol, cax=cax)
cbar.set_label(r'$\mathbf{MADN}$', fontsize=20, rotation=270, labelpad=30)

#====================================================================
#plt.show()
#====================================================================
if ("Trans." in model_type):
    model_type = model_type[:-1]

# fig_map.savefig(os.path.join('/Users/joshuamiller/Documents/Lancaster/Figs/MADNMaps', 
#                               'MADN_Pred_'+model_name+'_'+date_str+'.pdf'),
#                 bbox_inches='tight', pad_inches=0)

# fig_data.savefig(os.path.join('/Users/joshuamiller/Documents/Lancaster/Figs/MADNMaps', 
#                               'MADN_Data_'+model_type+'_'+date_str+'.pdf'),
#                 bbox_inches='tight', pad_inches=0)

# fig_diff.savefig(os.path.join('/Users/joshuamiller/Documents/Lancaster/Figs/MADNMaps', 
#                               'Diff_'+model_type+'_'+date_str+'.pdf'),
#                 bbox_inches='tight', pad_inches=0)

# fig_bar.savefig(os.path.join('/Users/joshuamiller/Documents/Lancaster/Figs/MADNMaps', 
#                               'MADNcolorbar.pdf'),
#                 bbox_inches='tight', pad_inches=0)