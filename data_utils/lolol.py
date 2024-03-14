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
from data_utils.extraction_funcs import Extract_netCDF4, ExtractHDF5, ExtractHDF5_v2
from misc.misc_utils import Longitude360to180, DownSample, PlotBoxes
#====================================================================
''' World map '''
world = gpd.read_file("/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")
crs = world.crs
print(" + + +", crs, crs.datum)

downsample_rate = 50

#====================================================================
fig, ax = plt.subplots(3, 1, figsize=(10,6)) # figsize (height (in), width (in))
#====================================================================
file = "/Users/joshuamiller/Documents/Lancaster/Data/ncdc_oisst_v2_avhrr_by_time_zlev_lat_lon_2430_0725_78b7.csv"
#file = "/Users/joshuamiller/Documents/Lancaster/Data/ncdc_oisst_v2_avhrr_by_time_zlev_lat_lon_a0fa_be1c_68b6.csv"
sst_df = pd.read_csv(file)

print("ADOJASHFA:SFLJAS:LFKJAS:FKJAS:FKAJS")
#print(sst_df.keys(), sst_df.head())
print("OEJWHRKAJSNFASOIWQKN ASEOHRASJKNFOA")

sst = sst_df['sst (Celsius)'].values
lat = sst_df['latitude (degrees_north)']
lon = Longitude360to180(sst_df['longitude (degrees_east)'])

sst = DownSample(sst, downsample_rate, 0, delete=True)
lat = DownSample(lat, downsample_rate, 0, delete=True)
lon = DownSample(lon, downsample_rate, 0, delete=True)
print(np.shape(sst), np.shape(lon), np.shape(lat))
#====================================================================
file = "/Users/joshuamiller/Documents/Lancaster/Data/NETCDF4_LSASAF_M01-AVHR_EDLST-DAY_GLOBE_202112310000.nc"
dict_ = Extract_netCDF4(file,
                        var_names=['lat', 'lon', 'time', 'aquisition_time-day', 'LST-day', 'VZA-day'],
                        groups='all',
                        print_sum=True)

aq_time = dict_['aquisition_time-day']
time    = dict_['time']
lon_lst = dict_['lon']
lat_lst = dict_['lat']
lst     = dict_['VZA-day'][0, :, :]
del(dict_)
#print('time=', time, ', aq_time=', aq_time)
lat_tiled = np.tile(lat_lst, (np.shape(lon_lst)[0], 1)).T
lon_tiled = np.tile(lon_lst, (np.shape(lat_lst)[0], 1))
del(lat_lst)
del(lon_lst)

lat_tiled = DownSample(DownSample(lat_tiled, downsample_rate, 0), downsample_rate, 1, delete=True)
lon_tiled = DownSample(DownSample(lon_tiled, downsample_rate, 0), downsample_rate, 1, delete=True)
lst = DownSample(DownSample(lst, downsample_rate, 0), downsample_rate, 1, delete=True)
#====================================================================
print("============================================================ ")
# file = "/Users/joshuamiller/Documents/Lancaster/Data/HDF5_LSASAF_M01-AVHR_EDLST-DAY_GLOBE_202212310000"
# dict_ = ExtractHDF5_v2(file,
#                     var_names=['LST-day', 'VZA-day'],
#                     groups='all',
#                     print_sum=True)


# aq_time = dict_['aquisition_time-day']
# time    = dict_['time']
# lon_lst = dict_['lon']
# lat_lst = dict_['lat']
# lst     = dict_['VZA-day'][0, :, :]
# del(dict_)
# #print('time=', time, ', aq_time=', aq_time)
# lat_tiled = np.tile(lat_lst, (np.shape(lon_lst)[0], 1)).T
# lon_tiled = np.tile(lon_lst, (np.shape(lat_lst)[0], 1))
# del(lat_lst)
# del(lon_lst)

# lat_tiled = DownSample(DownSample(lat_tiled, downsample_rate, 0), downsample_rate, 1, delete=True)
# lon_tiled = DownSample(DownSample(lon_tiled, downsample_rate, 0), downsample_rate, 1, delete=True)
# lst = DownSample(DownSample(lst, downsample_rate, 0), downsample_rate, 1, delete=True)
#====================================================================
# file = "/Users/joshuamiller/Documents/Lancaster/Data/NETCDF4_LSASAF_M01-AVHR_EDLST-DAY_GLOBE_202112310000.nc"
# dict_ = Extract_netCDF4(file,
#                         var_names=['lat', 'lon', 'time', 'aquisition_time-day', 'LST-day', 'VZA-day'],
#                         groups='all',
#                         print_sum=True)

# aq_time = dict_['aquisition_time-day']
# time    = dict_['time']
# lon_lst = dict_['lon']
# lat_lst = dict_['lat']
# lst     = dict_['VZA-day'][0, :, :]
# del(dict_)
# #print('time=', time, ', aq_time=', aq_time)
# lat_tiled = np.tile(lat_lst, (np.shape(lon_lst)[0], 1)).T
# lon_tiled = np.tile(lon_lst, (np.shape(lat_lst)[0], 1))
# del(lat_lst)
# del(lon_lst)

# lat_tiled = DownSample(DownSample(lat_tiled, downsample_rate, 0), downsample_rate, 1, delete=True)
# lon_tiled = DownSample(DownSample(lon_tiled, downsample_rate, 0), downsample_rate, 1, delete=True)
# lst = DownSample(DownSample(lst, downsample_rate, 0), downsample_rate, 1, delete=True)
#====================================================================
sst_norm = Normalize(vmin=min(sst), vmax=max(sst))
sst_cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'deepskyblue', 'lavenderblush', 'pink', 'red'], N=200) #8 Higher N=more smooth

lst_norm = Normalize(vmin=min(lst.ravel()), vmax=max(lst.ravel()))
lst_cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'deepskyblue', 'lavenderblush', 'pink', 'red'], N=200) #8 Higher N=more smooth
#====================================================================
sst_plot = ax[0].scatter(x=lon, y=lat, c=sst, 
                         s=5, marker='o', norm=sst_norm, cmap=sst_cmap)
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(sst_plot, cax=cax)
cbar.set_label(label='Temp. (C)', weight='bold', fontsize=16)
cbar.ax.tick_params(labelsize=20)
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=90)



lst_plot = ax[1].scatter(x=lon_tiled, y=lat_tiled, c=lst, 
                         s=5, marker='o', norm=lst_norm, cmap=lst_cmap)
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(lst_plot, cax=cax)
cbar.set_label(label='Temp. (?)', weight='bold', fontsize=16)
cbar.ax.tick_params(labelsize=20)
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=90)
#==========================================================================================
world.plot(ax=ax[0], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
world.plot(ax=ax[1], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map

fig2, ax2 = plt.subplots(1,1,figsize=(8, 4))

world.plot(ax=ax2, facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
PlotBoxes("data_utils/data_utils_config.yml", ax2, plot_text=True)
ax2.set_xlim(-30, 70)
ax2.set_ylim(-30, 30)

plt.show()

fig2.savefig('/Users/joshuamiller/Documents/Lancaster/Figs/Boxes.pdf', bbox_inches='tight', pad_inches=0)