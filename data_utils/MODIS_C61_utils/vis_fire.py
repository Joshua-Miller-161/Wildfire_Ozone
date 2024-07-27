import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
from matplotlib.colors import Normalize, LogNorm
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Point
import pandas as pd
import numpy as np
import random
import sys
from datetime import datetime, timedelta

import re

sys.path.append(os.getcwd())
from data_utils.extract_netCDF4 import Extract_netCDF4
from misc.misc_utils import SplitDataFrame
#====================================================================
''' Get data '''
# path1 = "/Users/joshuamiller/Documents/Lancaster/Data/MODIS_C61/fire_archive_M-C61_401077.csv"
# df = pd.read_csv(path1)
# print("df=", df)

# new_dfs = SplitDataFrame(df, column="acq_date", 
#                         save_new_files=True,
#                         new_files_folder="/Users/joshuamiller/Documents/Lancaster/Data/MODIS_C61")
#====================================================================
num_rows = 3
num_cols = 3
fig, ax = plt.subplots(num_rows, num_cols, figsize=(9,7))
#====================================================================
''' World map '''
world = gpd.read_file("/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")
crs = world.crs
print(" + + +", crs, crs.datum)
#====================================================================

gdf = gpd.read_file("/Users/joshuamiller/Documents/Lancaster/Data/Fire-MODIS_C61/DL_FIRE_M-C61_421476/fire_archive_M-C61_421476.shp")
crs = gdf.crs
print(" + + +", crs, crs.datum)

#====================================================================
path = "/Users/joshuamiller/Documents/Lancaster/Data/Fire-MODIS_C61"

files = os.listdir(path)
for file in files:
    if not file.endswith('.csv'):
        files.remove(file)

files_to_plot = random.sample(files, num_rows * num_cols)
#====================================================================
''' Make points '''
dfs_dict = {}
min_frp = 999
max_frp = -999
for i in range(num_rows):
    for j in range(num_cols):
        df = pd.read_csv(os.path.join(path, files_to_plot[num_cols * i + j]))
        #------------------------------------------------------------
        points = [Point(x,y) for x,y in zip(df['longitude'].values, df['latitude'].values)]
        points_gdf = gpd.GeoDataFrame(geometry=points)

        #------------------------------------------------------------
        ''' Fire dataframe'''
        fire_gdf = gpd.GeoDataFrame(geometry=points).assign(data=df['frp'].values)

        #------------------------------------------------------------
        if (max(df['frp'].values.ravel()) > max_frp):
            max_frp = max(df['frp'].values)

        if (min(df['frp'].values.ravel()) < min_frp):
            min_frp = min(df['frp'].values)

        #------------------------------------------------------------
        dfs_dict[files_to_plot[num_cols * i + j]] = fire_gdf

#====================================================================
print("min:", min_frp, ", max:", max_frp)
if (min_frp <= 0):
    min_frp = 10**-1
fire_norm = LogNorm(vmin=min_frp, vmax=max_frp)
fire_cmap = LinearSegmentedColormap.from_list('custom', ['yellow', 'orange', 'red'], N=200) # Higher N=more smooth

#====================================================================
for i in range(num_rows):
    for j in range(num_cols):
        #------------------------------------------------------------
        ''' Plot fire '''
        dfs_dict[files_to_plot[num_cols * i + j]].plot(ax=ax[i][j], column='data', cmap=fire_cmap, norm=fire_norm, markersize=.1, alpha=1, legend=True)
        world.plot(ax=ax[i][j], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map

        #------------------------------------------------------------
        ax[i][j].set_xlim(-20, 60)
        ax[i][j].set_ylim(-20, 20)
        ax[i][j].set_title(files_to_plot[num_cols * i + j])
#====================================================================
f = ['/Users/joshuamiller/Documents/Lancaster/Data/Fire-MODIS_C61/2018-07-08.csv',
     '/Users/joshuamiller/Documents/Lancaster/Data/Fire-MODIS_C61/2021-05-10.csv']

conf_norm = Normalize(vmin=0, vmax=100)
conf_cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'deepskyblue', 'lavenderblush', 'pink', 'red'], N=5) # Higher N=more smooth

fig2, ax2 = plt.subplots(1, 2, figsize=(12,6))
fig2.subplots_adjust(wspace=.5, hspace=.5)

file = 0
df = pd.read_csv(f[file])

fire = ax2[0].scatter(x=df.loc[:, 'longitude'], y=df.loc[:, 'latitude'], c=df.loc[:, 'frp'], s=2, cmap=fire_cmap, norm=fire_norm)

divider = make_axes_locatable(ax2[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(fire, cax=cax, label='Fire Radiative Potential (MW)')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=180)
#----------------------------------------------------------------
conf = ax2[1].scatter(x=df.loc[:, 'longitude'], y=df.loc[:, 'latitude'], c=df.loc[:, 'confidence'], s=2, cmap=conf_cmap, norm=conf_norm)

divider = make_axes_locatable(ax2[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(conf, cax=cax, label='Confidence (%)')

for i in range(2):
    world.plot(ax=ax2[i], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
    ax2[i].set_xlim(-20, 60)
    ax2[i].set_ylim(-20, 20)
    ax2[i].set_title(f[file][-14:-4])

#fig2.savefig(os.path.join("/Users/joshuamiller/Documents/Lancaster/Figs", "FireConf.pdf"), bbox_inches='tight', pad_inches=0)
#====================================================================
f_new = ['/Users/joshuamiller/Documents/Lancaster/Data/Kriged_MODIS_C61/MODIS_C61_2018-07-08_kriged.csv',
         '/Users/joshuamiller/Documents/Lancaster/Data/Kriged_MODIS_C61/MODIS_C61_2022-12-31_kriged.csv']
f_old = ['/Users/joshuamiller/Documents/Lancaster/Data/Fire-MODIS_C61/2018-07-08.csv',
         '/Users/joshuamiller/Documents/Lancaster/Data/Fire-MODIS_C61/2022-12-31.csv']
# search_folder = '/Users/joshuamiller/Documents/Lancaster/Data/Fire-MODIS_C61'
# pattern = r'\d{4}-\d{2}-\d{2}'
# for file in f_new:
#     for old_file in os.listdir(search_folder):
#         if old_file.endswith('csv'):
#             match = re.search(pattern, file)
#             if match:
#                 print("Date found:", match.group(0))
#                 f_old.append(os.path.join(search_folder, old_file))
#                 break

fig3, ax3 = plt.subplots(len(f_new), 2, figsize=(9,7))
fig3.subplots_adjust(wspace=.5, hspace=.5)

for i in range(len(f)):
    new_fire_df = pd.read_csv(f_new[i])
    old_fire_df = pd.read_csv(f_old[i])

    new_fire = ax3[i][0].scatter(x=new_fire_df.loc[:, 'lon'], y=new_fire_df.loc[:, 'lat'], c=new_fire_df.loc[:, 'frp'], s=1, cmap=fire_cmap, norm=fire_norm)
    
    divider = make_axes_locatable(ax3[i][0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(new_fire, cax=cax, label='Megatwatts???')
    cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=180)

    old_fire = ax3[i][1].scatter(x=old_fire_df.loc[:, 'longitude'], y=old_fire_df.loc[:, 'latitude'], c=old_fire_df.loc[:, 'frp'], s=1, cmap=fire_cmap, norm=fire_norm)
    
    divider = make_axes_locatable(ax3[i][1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(old_fire, cax=cax, label='Megatwatts???')
    cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=180)

for i in range(len(f)):
    for j in range(2):
        world.plot(ax=ax3[i][j], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
        ax3[i][j].set_xlim(-20, 60)
        ax3[i][j].set_ylim(-20, 20)
        ax3[i][j].set_title(f_new[i][-21:-11])
#====================================================================
fig4, ax4 = plt.subplots(2, 2, figsize=(12,8))
region = 'West_Ocean'
f_orig = '/Users/joshuamiller/Documents/Lancaster/Data/'+region+'/Fire/MODIS_C61_2018-05-24_'+region+'.csv'
f_nc   = '/Users/joshuamiller/Documents/Lancaster/Data/'+region+'/Fire/MODIS_C61_2018-04-30_2022-07-31_'+region+'.nc'
date_idx = 24

df_orig   = pd.read_csv(f_orig)

dict_nc = Extract_netCDF4(f_nc,
                          ['lat', 'lon', 'frp', 'date', 'confidence'],
                          groups='all',
                          print_sum=True)
lat_nc   = dict_nc['lat']
lon_nc   = dict_nc['lon']
dates_nc = dict_nc['date']
frp_nc   = dict_nc['frp'][date_idx, :, :]
conf_nc   = dict_nc['confidence'][date_idx, :, :]
print(dates_nc)

lat_nc_tiled = np.tile(lat_nc, (np.shape(lon_nc)[0], 1)).T
lon_nc_tiled = np.tile(lon_nc, (np.shape(lat_nc)[0], 1))
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
fire_orig = ax4[0][0].scatter(x=df_orig.loc[:, 'lon'], y=df_orig.loc[:, 'lat'], c=df_orig.loc[:, 'frp'], s=2, cmap=fire_cmap, norm=fire_norm)
divider = make_axes_locatable(ax4[0][0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(fire_orig, cax=cax, label='Megatwatts')

fire_nc = ax4[1][0].scatter(x=lon_nc_tiled, y=lat_nc_tiled, c=frp_nc, s=2, cmap=fire_cmap, norm=fire_norm)
divider = make_axes_locatable(ax4[1][0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(fire_nc, cax=cax, label='Megatwatts')

conf_orig = ax4[0][1].scatter(x=df_orig.loc[:, 'lon'], y=df_orig.loc[:, 'lat'], c=df_orig.loc[:, 'confidence'], s=2, cmap=conf_cmap, norm=conf_norm)

conf_nc   = ax4[1][1].scatter(x=lon_nc_tiled, y=lat_nc_tiled, c=conf_nc, s=2, cmap=conf_cmap, norm=conf_norm)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
for i in range(2):
    for j in range(2):
        world.plot(ax=ax4[i][j], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
        ax4[i][j].set_xlim(-21, 61)
        ax4[i][j].set_ylim(-21, 21)

ax4[0][0].set_title("csv "+f_orig[-25:-15])
ax4[1][0].set_title("nc "+datetime.strftime(datetime(1970, 1, 1)+timedelta(days=dates_nc[date_idx]), '%Y-%m-%d'))
#====================================================================
#====================================================================
#====================================================================
#====================================================================
#====================================================================
shape = gpd.read_file("/Users/joshuamiller/Desktop/DL_FIRE_M-C61_444221/fire_archive_M-C61_444221.shp")


print("====================================================================")
print("====================================================================")
print("====================================================================")
print("====================================================================")
print(shape.crs, shape.crs.datum)

print("====================================================================")
print("====================================================================")
print("====================================================================")
print("====================================================================")
#====================================================================
#====================================================================
#====================================================================
#====================================================================
#====================================================================
plt.show()