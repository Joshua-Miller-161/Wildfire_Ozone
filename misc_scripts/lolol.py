import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
from matplotlib.colors import Normalize, LogNorm, FuncNorm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import pandas as pd
import sys
import numpy as np
from datetime import datetime, timedelta
import yaml
import netCDF4 as nc
    
sys.path.append(os.getcwd())
from data_utils.extraction_funcs import Extract_netCDF4
from vis.plotting_utils import PlotBoxes, DegreeFormatter
#====================================================================
''' World map '''
world = gpd.read_file("/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")
crs = world.crs
print(" + + +", crs, crs.datum)
#====================================================================
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
fig.subplots_adjust(wspace=.3)
region = 'North_Land'

file_orig = '/Users/joshuamiller/Documents/Lancaster/Data/Whole_Area/Ozone/S5P_RPRO_L2__O3_TCL_2018-04-30_2022-07-31_Whole_Area.nc'
file_new = '/Users/joshuamiller/Documents/Lancaster/Data/'+region+'/Ozone/S5P_RPRO_L2__O3_TCL_2018-04-30_2022-07-31_'+region+'.nc'
date_idx = 100
var_to_plot = 'ozone_tropospheric_vertical_column'

# dataset = nc.Dataset(file_orig, 'r+')

# lon = dataset.variables['longitude'][:]
# print(min(lon), max(lon))
# if (max(lon)>180):
#     print('alllllsdasdas')
#     dataset.variables['lon'][:] = lon-360
# print(lon)
# dataset.close()

# dataset = nc.Dataset(file_orig, 'r')
# lon = dataset.variables['lon'][:]
# print("=============")
# print("new_lon =", lon)
# dataset.close()



dict_orig = Extract_netCDF4(file_orig,
                            var_names=['lon', 'lat', 'start_date', var_to_plot],
                            groups='all',
                            print_sum=True)

lon_orig  = dict_orig['lon']
lat_orig  = dict_orig['lat']
hours_orig = dict_orig['start_date']
data_orig = np.squeeze(dict_orig[var_to_plot])
lat_orig_tiled = np.tile(lat_orig, (np.shape(lon_orig)[0], 1)).T
lon_orig_tiled = np.tile(lon_orig, (np.shape(lat_orig)[0], 1))

min_ = min(data_orig.ravel())
max_ = max(data_orig.ravel())

# if (min_ <= 0):
#     min_ = 0.01
# norm = LogNorm(vmin=min_, vmax=max_)
norm = Normalize(vmin=min_, vmax=max_)
cmap = LinearSegmentedColormap.from_list('custom', ['blue',
                                                    'cornflowerblue',
                                                    'powderblue',
                                                    'pink',
                                                    'palevioletred',
                                                    'red'], N=200) # Higher N=more smooth

scat = ax[0].scatter(x=lon_orig_tiled, y=lat_orig_tiled, c=data_orig[date_idx, :, :], s=2, cmap=cmap, norm=norm)
world.plot(ax=ax[0], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
PlotBoxes("data_utils/data_utils_config.yml", ax[0], plot_text=True)

divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(scat, cax=cax, label='')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=180)

curr_date = datetime(1970, 1, 1) + timedelta(days=hours_orig[date_idx])
ax[0].set_title(curr_date.strftime('%Y-%m-%d %H:%M:%S'))
#--------------------------------------------------------------------
print("============================================================")
print("============================================================")
print("============================================================")
dict_new = Extract_netCDF4(file_new,
                           var_names=['lat', 'lon', 'start_date', var_to_plot],
                           groups='all',
                           print_sum=True)

lon_new   = dict_new['lon']
lat_new   = dict_new['lat']
hours_new = dict_new['start_date']
data_new  = dict_new[var_to_plot][date_idx, :, :]

lat_new_tiled = np.tile(lat_new, (np.shape(lon_new)[0], 1)).T
lon_new_tiled = np.tile(lon_new, (np.shape(lat_new)[0], 1))

scat_ = ax[1].scatter(x=lon_new_tiled, y=lat_new_tiled, c=data_new, s=2, cmap=cmap, norm=norm)
world.plot(ax=ax[1], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
PlotBoxes("data_utils/data_utils_config.yml", ax[1], plot_text=True)

divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(scat_, cax=cax, label='')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=180)

curr_date = datetime(1970, 1, 1) + timedelta(days=hours_new[date_idx])
ax[1].set_title(curr_date.strftime('%Y-%m-%d %H:%M:%S'))
#--------------------------------------------------------------------
ax[0].set_xlim(-21, 61)
ax[0].set_ylim(-21, 21)
ax[1].set_xlim(-21, 61)
ax[1].set_ylim(-21, 21)
plt.show()
















# #====================================================================
# fig, ax = plt.subplots(3, 1, figsize=(10,6)) # figsize (height (in), width (in))
# #====================================================================

# file = "/Users/joshuamiller/Documents/Lancaster/Data/ncdc_oisst_v2_avhrr_by_time_zlev_lat_lon_2430_0725_78b7.csv"
# #file = "/Users/joshuamiller/Documents/Lancaster/Data/ncdc_oisst_v2_avhrr_by_time_zlev_lat_lon_a0fa_be1c_68b6.csv"
# sst_df = pd.read_csv(file)

# print("ADOJASHFA:SFLJAS:LFKJAS:FKJAS:FKAJS")
# #print(sst_df.keys(), sst_df.head())
# print("OEJWHRKAJSNFASOIWQKN ASEOHRASJKNFOA")

# sst = sst_df['sst (Celsius)'].values
# lat = sst_df['latitude (degrees_north)']
# lon = Longitude360to180(sst_df['longitude (degrees_east)'])

# sst = DownSample(sst, downsample_rate, 0, delete=True)
# lat = DownSample(lat, downsample_rate, 0, delete=True)
# lon = DownSample(lon, downsample_rate, 0, delete=True)
# print(np.shape(sst), np.shape(lon), np.shape(lat))
# #====================================================================
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
# #====================================================================
# print("============================================================ ")
# # file = "/Users/joshuamiller/Documents/Lancaster/Data/HDF5_LSASAF_M01-AVHR_EDLST-DAY_GLOBE_202212310000"
# # dict_ = ExtractHDF5_v2(file,
# #                     var_names=['LST-day', 'VZA-day'],
# #                     groups='all',
# #                     print_sum=True)


# # aq_time = dict_['aquisition_time-day']
# # time    = dict_['time']
# # lon_lst = dict_['lon']
# # lat_lst = dict_['lat']
# # lst     = dict_['VZA-day'][0, :, :]
# # del(dict_)
# # #print('time=', time, ', aq_time=', aq_time)
# # lat_tiled = np.tile(lat_lst, (np.shape(lon_lst)[0], 1)).T
# # lon_tiled = np.tile(lon_lst, (np.shape(lat_lst)[0], 1))
# # del(lat_lst)
# # del(lon_lst)

# # lat_tiled = DownSample(DownSample(lat_tiled, downsample_rate, 0), downsample_rate, 1, delete=True)
# # lon_tiled = DownSample(DownSample(lon_tiled, downsample_rate, 0), downsample_rate, 1, delete=True)
# # lst = DownSample(DownSample(lst, downsample_rate, 0), downsample_rate, 1, delete=True)
# #====================================================================
# # file = "/Users/joshuamiller/Documents/Lancaster/Data/NETCDF4_LSASAF_M01-AVHR_EDLST-DAY_GLOBE_202112310000.nc"
# # dict_ = Extract_netCDF4(file,
# #                         var_names=['lat', 'lon', 'time', 'aquisition_time-day', 'LST-day', 'VZA-day'],
# #                         groups='all',
# #                         print_sum=True)

# # aq_time = dict_['aquisition_time-day']
# # time    = dict_['time']
# # lon_lst = dict_['lon']
# # lat_lst = dict_['lat']
# # lst     = dict_['VZA-day'][0, :, :]
# # del(dict_)
# # #print('time=', time, ', aq_time=', aq_time)
# # lat_tiled = np.tile(lat_lst, (np.shape(lon_lst)[0], 1)).T
# # lon_tiled = np.tile(lon_lst, (np.shape(lat_lst)[0], 1))
# # del(lat_lst)
# # del(lon_lst)

# # lat_tiled = DownSample(DownSample(lat_tiled, downsample_rate, 0), downsample_rate, 1, delete=True)
# # lon_tiled = DownSample(DownSample(lon_tiled, downsample_rate, 0), downsample_rate, 1, delete=True)
# # lst = DownSample(DownSample(lst, downsample_rate, 0), downsample_rate, 1, delete=True)
# #====================================================================
# sst_norm = Normalize(vmin=min(sst), vmax=max(sst))
# sst_cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'deepskyblue', 'lavenderblush', 'pink', 'red'], N=200) #8 Higher N=more smooth

# lst_norm = Normalize(vmin=min(lst.ravel()), vmax=max(lst.ravel()))
# lst_cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'deepskyblue', 'lavenderblush', 'pink', 'red'], N=200) #8 Higher N=more smooth
# #====================================================================
# sst_plot = ax[0].scatter(x=lon, y=lat, c=sst, 
#                          s=5, marker='o', norm=sst_norm, cmap=sst_cmap)
# divider = make_axes_locatable(ax[0])
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cbar = plt.colorbar(sst_plot, cax=cax)
# cbar.set_label(label='Temp. (C)', weight='bold', fontsize=16)
# cbar.ax.tick_params(labelsize=20)
# cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=90)



# lst_plot = ax[1].scatter(x=lon_tiled, y=lat_tiled, c=lst, 
#                          s=5, marker='o', norm=lst_norm, cmap=lst_cmap)
# divider = make_axes_locatable(ax[1])
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cbar = plt.colorbar(lst_plot, cax=cax)
# cbar.set_label(label='Temp. (?)', weight='bold', fontsize=16)
# cbar.ax.tick_params(labelsize=20)
# cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=90)
#==========================================================================================
fig2, ax2 = plt.subplots(1,1,figsize=(8, 4))

world.plot(ax=ax2, facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
PlotBoxes("data_utils/data_utils_config.yml", ax2, plot_text=True, fontsize=16)

ax2.set_xlim(-22, 62)
ax2.set_ylim(-22, 22)

ax2.xaxis.set_major_formatter(FuncFormatter(DegreeFormatter))
ax2.yaxis.set_major_formatter(FuncFormatter(DegreeFormatter))

plt.show()

#fig2.savefig('/Users/joshuamiller/Documents/Lancaster/Figs/Boxes.pdf', bbox_inches='tight', pad_inches=0)