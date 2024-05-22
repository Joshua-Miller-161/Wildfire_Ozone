import sys
sys.dont_write_bytecode = True
import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
from matplotlib.colors import Normalize, LogNorm, FuncNorm
from matplotlib.colors import LinearSegmentedColormap
import sys
import numpy as np
from datetime import datetime, timedelta
    
sys.path.append(os.getcwd())
from data_utils.extraction_funcs import Extract_netCDF4
from misc.plotting_utils import PlotBoxes
#====================================================================
''' World map '''
world = gpd.read_file("/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")
crs = world.crs
print(" + + +", crs, crs.datum)
#====================================================================
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
ozone_file = '/Users/joshuamiller/Documents/Lancaster/Data/Whole_Area/Ozone/S5P_RPRO_L2__O3_TCL_2018-04-30_2022-07-31_Whole_Area.nc'
date_idx = 160

dict_orig = Extract_netCDF4(ozone_file,
                            var_names=['lon', 'lat', 'start_date', 'ozone_tropospheric_vertical_column'],
                            groups='all',
                            print_sum=True)

lon_orig  = dict_orig['lon']
lat_orig  = dict_orig['lat']
hours_orig = dict_orig['start_date']
data_orig = np.squeeze(dict_orig['ozone_tropospheric_vertical_column'])
lat_orig_tiled = np.tile(lat_orig, (np.shape(lon_orig)[0], 1)).T
lon_orig_tiled = np.tile(lon_orig, (np.shape(lat_orig)[0], 1))

max_ = max(data_orig.ravel())
norm = Normalize(0, max_)

scat = ax1.scatter(x=lon_orig_tiled, y=lat_orig_tiled, c=data_orig[date_idx, :, :], s=10, cmap='bwr', norm=norm)
world.plot(ax=ax1, facecolor='none', edgecolor='black', linewidth=.5, alpha=1) # GOOD lots the map
PlotBoxes("data_utils/data_utils_config.yml", ax1, plot_text=False)

divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(scat, cax=cax)
#cbar.set_label(label='Ozone (mol/m^2)', fontsize=16, weight='bold', rotation=270)

curr_date = datetime(1970, 1, 1) + timedelta(days=hours_orig[date_idx])
#ax1.set_title(curr_date.strftime('%Y-%m-%d %H:%M:%S'))
#--------------------------------------------------------------------
ax1.set_xlim(-21, 61)
ax1.set_ylim(-21, 21)
ax1.set_xlim(-21, 61)
ax1.set_ylim(-21, 21)

ax1.set_xticks([])
ax1.set_yticks([])
print("============================================================")
print("============================================================")
print("============================================================")
#====================================================================
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
fire_file = '/Users/joshuamiller/Documents/Lancaster/Data/Whole_Area/Fire/MODIS_C61_2018-04-30_2022-07-31_Whole_Area.nc'

dict_new = Extract_netCDF4(fire_file,
                           var_names=['lat', 'lon', 'date', 'frp'],
                           groups='all',
                           print_sum=True)

lon_new   = dict_new['lon']
lat_new   = dict_new['lat']
days_new = dict_new['date']
data_new  = dict_new['frp'][date_idx, :, :]

lat_fire_tiled = np.tile(lat_new, (np.shape(lon_new)[0], 1)).T
lon_fire_tiled = np.tile(lon_new, (np.shape(lat_new)[0], 1))

valid_lon = []
valid_lat = []
valid_fire = []

for i in range(np.shape(data_new)[0]):
    for j in range(np.shape(data_new)[1]):
        if (data_new[i][j]>0):
            valid_lon.append(lon_fire_tiled[i][j])
            valid_lat.append(lat_fire_tiled[i][j])
            valid_fire.append(data_new[i][j])

fire_norm = LogNorm(1, 10**3)
fire_cmap = LinearSegmentedColormap.from_list('custom', ['yellow', 'orange', 'red'], N=200)

scat_ = ax2.scatter(x=valid_lon, y=valid_lat, c=valid_fire, s=10, cmap=fire_cmap, norm=fire_norm)
world.plot(ax=ax2, facecolor='none', edgecolor='black', linewidth=.5, alpha=1) # GOOD lots the map
PlotBoxes("data_utils/data_utils_config.yml", ax2, plot_text=False)

divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(scat_, cax=cax)
#cbar.set_label(label='Fire (MW)', fontsize=16, weight='bold', rotation=270)

curr_date = datetime(1970, 1, 1) + timedelta(days=days_new[date_idx])
#ax2.set_title(curr_date.strftime('%Y-%m-%d %H:%M:%S'))
#--------------------------------------------------------------------
ax2.set_xlim(-21, 61)
ax2.set_ylim(-21, 21)
ax2.set_xlim(-21, 61)
ax2.set_ylim(-21, 21)

ax2.set_xticks([])
ax2.set_yticks([])

fig1.savefig(os.path.join('Figs', 'Ozone.pdf'), bbox_inches='tight', pad_inches=0)
fig2.savefig(os.path.join('Figs', 'Fire.pdf'), bbox_inches='tight', pad_inches=0)
#====================================================================
plt.show()