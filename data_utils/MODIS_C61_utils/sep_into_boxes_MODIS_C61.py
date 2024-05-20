import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
from matplotlib.colors import Normalize, LogNorm, FuncNorm
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import sys
import numpy as np
import yaml
    
sys.path.append(os.getcwd())
from data_utils.extraction_funcs import Extract_netCDF4
from misc.misc_utils import GetBoxCoords, PlotBoxes
#====================================================================
save_folder = '/Users/joshuamiller/Documents/Lancaster/Data/'
data_folder = '/Users/joshuamiller/Documents/Lancaster/Data/Kriged_MODIS_C61'
#====================================================================
boxes = GetBoxCoords("data_utils/data_utils_config.yml")
#====================================================================
# ''' Filter into the boxes '''
# for file in os.listdir(data_folder):
#     if file.endswith('.csv'):
#         #------------------------------------------------------------
#         df = pd.read_csv(os.path.join(data_folder, file))
        
#         #------------------------------------------------------------
#         for box_name in boxes.keys():
#             #print(" >> Searching in", box_name)
#             sub_df = df.loc[(boxes[box_name][3] <= df['lat']) & (df['lat'] <= boxes[box_name][1]) & (boxes[box_name][0] <= df['lon']) & (df['lon'] <= boxes[box_name][2])]
            
#             path = save_folder+'/'+box_name+'/'+'MODIS_C61'
#             filename = file[:20]+'_'+box_name+'.csv'
#             #print(filename)
#             sub_df.to_csv(os.path.join(path, filename), index=False)
#             print("  >> Saved:", os.path.join(save_folder, box_name), ".csv |", str(sub_df.shape[0]), "values in box")
#====================================================================

fig, ax = plt.subplots(2, 1, figsize=(9,6)) # figsize (height (in), width (in))
fig.subplots_adjust(wspace=1, hspace=-.1)
#====================================================================
fire_norm = Normalize(vmin=0.1, vmax=10**3)
fire_cmap = LinearSegmentedColormap.from_list('custom', ['yellow', 'orange', 'red'], N=200) # Higher N=more smooth

#====================================================================

new_fire_df = pd.read_csv("/Users/joshuamiller/Documents/Lancaster/Data/Kriged_MODIS_C61/MODIS_C61_2018-04-29_kriged.csv")

frp = []
lat = []
lon = []
frp_0 = []
lat_0 = []
lon_0 = []
for j in range(new_fire_df.shape[0]):
    if (new_fire_df.loc[j, 'frp'] > 0.1):
        frp.append(new_fire_df.loc[j, 'frp'])
        lat.append(new_fire_df.loc[j, 'lat'])
        lon.append(new_fire_df.loc[j, 'lon'])
    else:
        frp_0.append(new_fire_df.loc[j, 'frp'])
        lat_0.append(new_fire_df.loc[j, 'lat'])
        lon_0.append(new_fire_df.loc[j, 'lon'])

new_fire = ax[0].scatter(x=lon, y=lat, c=frp, 
                         s=0.5, marker='.', cmap=fire_cmap, norm=fire_norm)
ax[0].scatter(x=lon_0, y=lat_0, s=0.5, marker='.', color='black')

new_fire_df = pd.read_csv("/Users/joshuamiller/Documents/Lancaster/Data/East_Ocean/MODIS_C61/MODIS_C61_2018-04-29_East_Ocean.csv")
frp = []
lat = []
lon = []
frp_0 = []
lat_0 = []
lon_0 = []
for j in range(new_fire_df.shape[0]):
    if (new_fire_df.loc[j, 'frp'] > 0.1):
        frp.append(new_fire_df.loc[j, 'frp'])
        lat.append(new_fire_df.loc[j, 'lat'])
        lon.append(new_fire_df.loc[j, 'lon'])
    else:
        frp_0.append(new_fire_df.loc[j, 'frp'])
        lat_0.append(new_fire_df.loc[j, 'lat'])
        lon_0.append(new_fire_df.loc[j, 'lon'])

new_fire = ax[1].scatter(x=lon, y=lat, c=frp, 
                         s=0.5, marker='.', cmap=fire_cmap, norm=fire_norm)
ax[1].scatter(x=lon_0, y=lat_0, s=0.5, marker='.', color='black')

world = gpd.read_file("/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")
world.plot(ax=ax[0], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
world.plot(ax=ax[1], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
PlotBoxes("data_utils/data_utils_config.yml", ax[0])
PlotBoxes("data_utils/data_utils_config.yml", ax[1])
ax[0].set_xlim(-30, 70)
ax[0].set_ylim(-30, 30)
ax[1].set_xlim(-30, 70)
ax[1].set_ylim(-30, 30)

plt.show()