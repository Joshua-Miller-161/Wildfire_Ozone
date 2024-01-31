import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
from matplotlib.colors import Normalize, LogNorm
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
from shapely.geometry import Point
import netCDF4 as nc
import numpy as np
import sys
import os

sys.path.append(os.getcwd())
from data_utils.preprocessing_funcs import Scale
from data_utils.extraction_funcs import Extract_netCDF4
#====================================================================
''' Make subplot '''
fig, ax = plt.subplots(3, 1, figsize=(8, 6))
fig.subplots_adjust(wspace=0, hspace=.5)

replace_val = 0.001
#====================================================================
''' Get data '''
folder_0 = "/Users/joshuamiller/Documents/Lancaster/Data/Ozone-L2_O3_TCL"
#f = "S5P_RPRO_L2__O3_TCL_20190219T115225_20190225T123845_07053_03_020401_20230329T133842/S5P_RPRO_L2__O3_TCL_20190219T115225_20190225T123845_07053_03_020401_20230329T133842.nc"
f = "S5P_RPRO_L2__O3_TCL_20190220T113329_20190226T121949_07067_03_020401_20230329T133844/S5P_RPRO_L2__O3_TCL_20190220T113329_20190226T121949_07067_03_020401_20230329T133844.nc"
#f = "S5P_RPRO_L2__O3_TCL_20190221T111433_20190227T115447_07081_03_020401_20230329T133845/S5P_RPRO_L2__O3_TCL_20190221T111433_20190227T115447_07081_03_020401_20230329T133845.nc"
#f = "S5P_RPRO_L2__O3_TCL_20190222T105538_20190228T114158_07095_03_020401_20230329T133847/S5P_RPRO_L2__O3_TCL_20190222T105538_20190228T114158_07095_03_020401_20230329T133847.nc"
dict_ = Extract_netCDF4(os.path.join(folder_0, f),
                        ['latitude_ccd', 'longitude_ccd', 'ozone_tropospheric_vertical_column', 'dates_for_tropospheric_column'],
                        groups='all',
                        print_sum=True)

lat_0 = dict_['latitude_ccd']
lon_0 = dict_['longitude_ccd']
ozone_0 = np.squeeze(dict_['ozone_tropospheric_vertical_column'])
ozone_0 = np.ma.getdata(np.squeeze(dict_['ozone_tropospheric_vertical_column']))
ozone_0 = np.where(ozone_0 > 100, replace_val, ozone_0)
ozone_0 = np.where(ozone_0 <= 0, 10**-10, ozone_0)

#print(np.shape(lon_0), np.shape(lat_0), np.shape(ozone_0))
#print("lon:", lon_0[0:3], ", lat:", lat_0[0:3], ", o3:", ozone_0[0:3, 0:3])

dates_0 = ''.join(dict_['dates_for_tropospheric_column'])
#--------------------------------------------------------------------
folder_1 = "/Users/joshuamiller/Documents/Lancaster/Data/Filtered_L2_O3_TCL"
for file_1 in os.listdir(folder_1):
    if file_1.endswith('.nc'):
        dataset = nc.Dataset(os.path.join(folder_1, file_1), 'r')
        if (f[-30:-1] in dataset.source):
            print("---")
            print("MATCH: dates_0: ", dates_0, "\n", dataset.source)
            print(type(folder_1), type(file_1), file_1)
            print("---")
            break

dict_ = Extract_netCDF4(os.path.join(folder_1, file_1),
                        ['lat', 'lon', 'ozone_tropospheric_vertical_column', 'dates_for_tropospheric_column'],
                        groups='all',
                        print_sum=False)

lat_1 = dict_['lat']
lon_1 = dict_['lon']
ozone_1 = np.squeeze(dict_['ozone_tropospheric_vertical_column'])
ozone_1 = np.ma.getdata(np.squeeze(dict_['ozone_tropospheric_vertical_column']))
ozone_1 = np.where(ozone_1 > 100, replace_val, ozone_1)
ozone_1 = np.nan_to_num(ozone_1, nan=replace_val)
ozone_1 = np.where(ozone_1 <= 0, 10**-10, ozone_1)

#print(np.shape(lon_1), np.shape(lat_1), np.shape(ozone_1))
#print("lon:", lon_1[0:3], ", lat:", lat_1[0:3], ", o3:", ozone_1[0:3, 0:3])
#print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
dates_1 = ''.join(dict_['dates_for_tropospheric_column'])

#--------------------------------------------------------------------
folder_3 = "/Users/joshuamiller/Documents/Lancaster/Data/Kriged_L2_O3_TCL"
for file_3 in os.listdir(folder_3):
    if file_3.endswith('.nc'):
        dataset = nc.Dataset(os.path.join(folder_3, file_3), 'r')
        if (file_1 in dataset.source):
            print("---")
            print("MATCH: dates_0: ", dates_0, "\n", dataset.source)
            print("---")
            break
        
dict_ = Extract_netCDF4(os.path.join(folder_3, file_3),
                        ['lat', 'lon', 'ozone_tropospheric_vertical_column', 'dates_for_tropospheric_column'],
                        groups='all',
                        print_sum=True)

lat_3 = dict_['lat']
lon_3 = dict_['lon']
ozone_3 = np.squeeze(dict_['ozone_tropospheric_vertical_column'])
dates_3 = ''.join(dict_['dates_for_tropospheric_column'])

#====================================================================
''' Plot ozone '''
date = 0

# - - - - - - - - - Get points for the ozone plot - - - - - - - - - - -
lat_tiled_0 = np.tile(lat_0, (np.shape(lon_0)[0], 1)).T
lon_tiled_0 = np.tile(lon_0, (np.shape(lat_0)[0], 1))
# print("00000000000")
# print("lat_tiled:", lat_tiled_0)
# print('00000000000')
# print("lon_tiled:", lon_tiled_0)
# print('00000000000')
lat_tiled_1 = np.tile(lat_1, (np.shape(lon_1)[0], 1)).T
lon_tiled_1 = np.tile(lon_1, (np.shape(lat_1)[0], 1))

latlon_points = np.array(list(zip(lat_tiled_1.ravel(), lon_tiled_1.ravel())))
O3_points = ozone_1.ravel()
print('00000000000')
print(np.shape(latlon_points), np.shape(O3_points), latlon_points[:5,:])



lat_tiled_3 = np.tile(lat_3, (np.shape(lon_3)[0], 1)).T
lon_tiled_3 = np.tile(lon_3, (np.shape(lat_3)[0], 1))
# - - - - - - - - - - - Make colorbar for ozone - - - - - - - - - - -
print(min(ozone_1.ravel()), min(ozone_0.ravel()))
# ozone_norm = Normalize(vmin=min([min(ozone_1.ravel()), min(ozone_0.ravel())]), 
#                        vmax=max([max(ozone_1.ravel()), max(ozone_0.ravel())]))
ozone_norm = Normalize(vmin=0, vmax=0.024)
ozone_cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'deepskyblue', 'lavenderblush', 'pink', 'red'], N=800) #8 Higher N=more smooth

# - - - - - - - - - - - - - Plot ozone data - - - - - - - - - - - - -
O3_0 = ax[0].scatter(x=lon_tiled_0, y=lat_tiled_0, c=ozone_0, 
                     s=2, marker='o', norm=ozone_norm, cmap=ozone_cmap)

# for i in range(np.shape(ozone_0)[0]): # Latitude dimension
#     for j in range(np.shape(ozone_0)[1]): # Longitude dimension
#         O3_0 = ax[0].scatter(x=lon_0[j], y=lat_0[i], c=ozone_0[i][j], 
#                              s=2, marker='o', norm=ozone_norm, cmap=ozone_cmap)

divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(O3_0, cax=cax, label=r'$\frac{mol}{m^2}$')



O3_1 = ax[1].scatter(x=lon_tiled_1, y=lat_tiled_1, c=ozone_1,
                     s=2, marker='o', norm=ozone_norm, cmap=ozone_cmap)
# for i in range(np.shape(ozone_1)[0]): # Latitude dimension
#     for j in range(np.shape(ozone_1)[1]): # Longitude dimension
#         O3_1 = ax[1].scatter(x=lon_1[j], y=lat_1[i], c=ozone_1[i][j], 
#                              s=2, marker='o', norm=ozone_norm, cmap=ozone_cmap)
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(O3_1, cax=cax, label=r'$\frac{mol}{m^2}$')



# O3_points = ax[2].scatter(x=latlon_points[:, 1], y=latlon_points[:, 0], c=O3_points,
#                           s=2, marker='o', norm=ozone_norm, cmap=ozone_cmap)

# divider = make_axes_locatable(ax[2])
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(O3_points, cax=cax, label=r'$\frac{mol}{m^2}$')


O3_3 = ax[2].scatter(x=lon_tiled_3, y=lat_tiled_3, c=ozone_3,
                     s=2, marker='o', norm=ozone_norm, cmap=ozone_cmap)

divider = make_axes_locatable(ax[2])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(O3_3, cax=cax, label=r'$\frac{mol}{m^2}$')
#====================================================================
''' Plot world map '''
world = gpd.read_file("/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")

world.plot(ax=ax[0], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
world.plot(ax=ax[1], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
world.plot(ax=ax[2], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map


# for i in range(3):
#     for j in range(3):
#         ax[0].text(lon_0[i], lat_0[j], str(round(ozone_0[j][i], 4)))
#         ax[1].text(lon_1[i], lat_1[j], str(round(ozone_1[j][i], 4)))
#====================================================================
print("================================")
print(dict_['dates_for_tropospheric_column'])
print(''.join(dict_['dates_for_tropospheric_column']))
ax[0].set_title('Dates: '+ dates_0, fontsize=6)
ax[1].set_title('Dates: '+ dates_1, fontsize=6)
ax[2].set_title('Dates: '+ dates_3, fontsize=6)
#====================================================================
plt.show()

# mask = np.ma.getmaskarray(dict_['ozone_tropospheric_vertical_column'])
# for i in range(np.shape(mask)[0]):
#     for j in range(np.shape(mask)[1]):
#         print("i =", i, ", j =", j, ", mask =", mask[i][j], ", val =", dict_['ozone_tropospheric_vertical_column'][i][j], type(dict_['ozone_tropospheric_vertical_column'][i][j]))