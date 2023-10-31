import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import Normalize, LogNorm
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Point
import pandas as pd
import os
import random
import sys

sys.path.append(os.getcwd())
from data_utils.extraction_funcs import SplitDataFrame
#====================================================================
''' Get data '''
# path1 = "/Users/joshuamiller/Documents/Lancaster/Data/MODIS_C61/fire_archive_M-C61_396750.csv"
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
#====================================================================
path = "/Users/joshuamiller/Documents/Lancaster/Data/MODIS_C61"

files = os.listdir(path)
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
fire_norm = LogNorm(vmin=min_frp, vmax=max_frp)
fire_cmap = LinearSegmentedColormap.from_list('custom', ['yellow', 'orange', 'red'], N=200) # Higher N=more smooth

#====================================================================
for i in range(num_rows):
    for j in range(num_cols):
        #------------------------------------------------------------
        ''' Plot fire '''
        dfs_dict[files_to_plot[num_cols * i + j]].plot(ax=ax[i][j], column='data', cmap=fire_cmap, norm=fire_norm, markersize=1, alpha=1, legend=True)
        world.plot(ax=ax[i][j], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map

        #------------------------------------------------------------
        ax[i][j].set_xlim(-20, 60)
        ax[i][j].set_ylim(-20, 20)
        ax[i][j].set_title(files_to_plot[num_cols * i + j])

plt.show()