import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
from shapely.geometry import Point
import netCDF4 as nc
import numpy as np
import pygrib
import iris_grib
import iris
import iris.quickplot as qplt
import re
import sys
import os
import cartopy
from cartopy import crs

sys.path.append(os.getcwd())
from preprocessing_funcs import Scale
from extraction_funcs import ExtractGRIB
from misc.misc_utils import FindDate
#====================================================================
''' Get data '''
dict_ = ExtractGRIB("/Users/joshuamiller/Documents/Lancaster/Data/Copernicus/adaptor.mars_constrained.external-1697707401.7930443-25417-7-468c83e5-7003-4cc1-b631-9d7c806b74c3.grib",
                    ['u-component of wind', 'v-component of wind'],
                    print_sum=True)


print("============================================================")
print("============================================================")
print("============================================================")
print("============================================================")
cubes = iris_grib.load_cubes("/Users/joshuamiller/Documents/Lancaster/Data/Copernicus/adaptor.mars_constrained.external-1697707401.7930443-25417-7-468c83e5-7003-4cc1-b631-9d7c806b74c3.grib")
cubes = list(cubes)
print("IRIS:", len(cubes), type(cubes[0]), vars(cubes[0])) 


cube_num = 3
print(cubes[cube_num]._dim_coords_and_dims[0][0], type(cubes[cube_num]._dim_coords_and_dims[0][0]))
print(cubes[cube_num].coord('latitude').points, type(cubes[cube_num].coord('latitude').points))
print(cubes[cube_num].coord('longitude').points - 360, type(cubes[cube_num].coord('longitude').points))
print(cubes[cube_num].coord('model_level_number').points, type(cubes[cube_num].coord('model_level_number').points))
print(cubes[cube_num].coord('time').points, type(cubes[cube_num].coord('time').points))
print(cubes[cube_num].coord('forecast_reference_time'), cubes[cube_num].coord('forecast_reference_time').points)


# # Get the string representation of the coordinate
# coord_str = str(cubes[cube_num].coord('forecast_reference_time'))

# # Extract the date string after the colon
# date_str = FindDate(coord_str, 'points')

# qplt.contourf(cubes[cube_num], 200)
# plt.gca().coastlines()
# plt.title(str(cubes[cube_num].standard_name)+', level: '+str(cubes[cube_num].coord('model_level_number').points[0])+', date: '+date_str)
# plt.show()



# Get the coordinate string
coord_str = str(cubes[cube_num].coord('forecast_reference_time'))

# Extract the date string after the colon
date_str = FindDate(coord_str, 'points')

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': crs.PlateCarree()})

# Plot the y-wind on the first subplot
qplt.contourf(cubes[cube_num], 200, axes=ax1)
ax1.coastlines()
ax1.set_title(str(cubes[cube_num].standard_name))

# Plot the x-wind on the second subplot
qplt.contourf(cubes[cube_num + 1], 200, axes=ax2)
ax2.coastlines()
ax2.set_title(str(cubes[cube_num + 1].standard_name))

# Show the figure
plt.suptitle('From Iris - Level: '+str(cubes[cube_num + 1].coord('model_level_number').points[0])+', Pa: '+str(round(cubes[cube_num + 1].coord('level_pressure').points[0],2))+', date: '+date_str)
plt.show()


#====================================================================


# ''' Make subplot '''
# fig, ax = plt.subplots(figsize=(8, 6))

# #ax.set_xlim(min(dict_['longitude_ccd']) - .1, max(dict_['longitude_ccd']) + .1)
# #ax.set_ylim(min(dict_['latitude_ccd']) - .1, max(dict_['latitude_ccd']) + .1)
# #====================================================================
# ''' Plot world map '''
# world = gpd.read_file("/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")

# world.plot(ax=ax, color='white', edgecolor='black', linewidth=0.1, alpha=1, legend=True) # GOOD lots the map
# #====================================================================
# ''' Plot ozone '''
# date = 0
# ozone = dict_['ozone_tropospheric_vertical_column'][date, :, :]

# # - - - - - - - - - Get points for the ozone plot - - - - - - - - - - -
# lat = np.tile(dict_['latitude_ccd'], (np.shape(dict_['longitude_ccd'])[0], 1)).T
# lon = np.tile(dict_['longitude_ccd'], (np.shape(dict_['latitude_ccd'])[0], 1))

# points = [Point(x,y) for x,y in zip(lon.ravel(), lat.ravel())]
# points_gdf = gpd.GeoDataFrame(geometry=points)

# print('lat:', np.shape(lat),
#       ', lon:', np.shape(lon),
#       ', u:', np.shape(u_wind),
#       ', v:', np.shape(v_wind))

# del(lat)
# del(lon)

# ozone_gdf = gpd.GeoDataFrame(geometry=points).assign(data=ozone.ravel())

# # - - - - - - - - - - - Make colorbar for ozone - - - - - - - - - - -
# ozone_norm = Normalize(vmin=0, vmax=max(dict_['ozone_tropospheric_vertical_column'].ravel()))
# ozone_cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'red'], N=200) # Higher N=more smooth

# # - - - - - - - - - - - - - Plot ozone data - - - - - - - - - - - - -
# ozone_gdf.plot(ax=ax, column='data', cmap=ozone_cmap, norm=ozone_norm, markersize=5, alpha=1, legend=True)
# #====================================================================
# plt.title(str(dict_['time']))

# #====================================================================
# plt.show()