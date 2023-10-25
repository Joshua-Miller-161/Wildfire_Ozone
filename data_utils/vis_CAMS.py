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
from pprint import pprint

sys.path.append(os.getcwd())
from preprocessing_funcs import Scale
from extraction_funcs import ExtractGRIBIris, PrintSumGRIBIris
from misc.misc_utils import FindDate
#====================================================================
''' Get data '''
dict_ = ExtractGRIBIris("/Users/joshuamiller/Documents/Lancaster/Data/Copernicus/adaptor.mars_constrained.external-1697707401.7930443-25417-7-468c83e5-7003-4cc1-b631-9d7c806b74c3.grib",
                        'all',
                        print_sum=True,
                        use_dask_array=False)

print("============================================================")
print("============================================================")
print("============================================================")
print("============================================================")
#print(dict_)
#====================================================================
#====================================================================
#====================================================================
#====================================================================
cubes = iris_grib.load_cubes("/Users/joshuamiller/Documents/Lancaster/Data/Copernicus/adaptor.mars_constrained.external-1697707401.7930443-25417-7-468c83e5-7003-4cc1-b631-9d7c806b74c3.grib")
cubes = list(cubes)
cube_num = 20
#PrintSumGRIBIris(cubes, 2)

# Get the coordinate string
coord_str = str(cubes[cube_num].coord('forecast_reference_time'))

# Extract the date string after the colon
date_str = FindDate(coord_str, 'points')

# Create a figure with two subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 5), subplot_kw={'projection': crs.PlateCarree()})

# Plot the y-wind on the first subplot
qplt.contourf(cubes[cube_num], 200, axes=ax1)
ax1.coastlines()
ax1.set_title('Iris - '+ str(cubes[cube_num].standard_name)+' Level: '+str(cubes[cube_num].coord('model_level_number').points[0])+', date: '+date_str)

# Plot the x-wind on the second subplot
qplt.contourf(cubes[cube_num + 1], 200, axes=ax2)
ax2.coastlines()
ax2.set_title('Iris - '+ str(cubes[cube_num + 1].standard_name)+' Level: '+str(cubes[cube_num + 1].coord('model_level_number').points[0])+', date: '+date_str)
#====================================================================
#====================================================================
#====================================================================
#====================================================================
''' Plot world map '''
world = gpd.read_file("/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")
#====================================================================
''' Plot wind '''

date_idx = 5
level_idx = 0
x_wind = dict_['x_wind'][date_idx, level_idx, :, :]
y_wind = dict_['y_wind'][date_idx, level_idx, :, :]
# - - - - - - - - - Get points for the ozone plot - - - - - - - - - - -
lat = np.tile(dict_['latitude'], (np.shape(dict_['longitude'])[0], 1)).T
lon = np.tile(dict_['longitude'], (np.shape(dict_['latitude'])[0], 1))

points = [Point(x,y) for x,y in zip(lon.ravel(), lat.ravel())]
points_gdf = gpd.GeoDataFrame(geometry=points)

del(lat)
del(lon)

x_wind_gdf = gpd.GeoDataFrame(geometry=points).assign(data=x_wind.ravel())
y_wind_gdf = gpd.GeoDataFrame(geometry=points).assign(data=y_wind.ravel())

# - - - - - - - - - - - Make wind for ozone - - - - - - - - - - -
x_wind_norm = Normalize(vmin=min(x_wind.ravel()), vmax=max(x_wind.ravel()))
x_wind_cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'green', 'yellow'], N=200) # Higher N=more smooth

y_wind_norm = Normalize(vmin=min(y_wind.ravel()), vmax=max(y_wind.ravel()))
y_wind_cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'green', 'yellow'], N=200) # Higher N=more smooth

# - - - - - - - - - - - - - Plot wind data - - - - - - - - - - - - -
x_wind_gdf.plot(ax=ax3, column='data', cmap='viridis', norm=x_wind_norm, markersize=1, alpha=1, legend=True)
y_wind_gdf.plot(ax=ax4, column='data', cmap='viridis', norm=y_wind_norm, markersize=1, alpha=1, legend=True)
#====================================================================
world.plot(ax=ax3, facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
world.plot(ax=ax4, facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
#====================================================================
ax3.set_title('Manual - x_wind, level: ' + str(dict_['model_level_number'][level_idx]) + ', date: ' + dict_['forecast_reference_time'][date_idx])
ax4.set_title('Manual - y_wind, level: ' + str(dict_['model_level_number'][level_idx]) + ', date: ' + dict_['forecast_reference_time'][date_idx])
#====================================================================
plt.show()





















# print("IRIS:", len(cubes), type(cubes[0])) 
# print("===========================================================")
# print("===========================================================")
# print("vars(cubes[0]):", vars(cubes[0]))
# print("===========================================================")
# print("===========================================================")
# print("cubes[0].__dir__:", cubes[0].__dir__())
# print("===========================================================")
# print("===========================================================")
# print("cubes[0].__dict__:", cubes[0].__dict__)
# print("===========================================================")
# print("===========================================================")
# print("cubes[0].data.__dir__():", cubes[0].units, cubes[0].data.__dir__())
# print("===========================================================")
# print("===========================================================")
# print("cubes[0].coord.__dir__():", cubes[0].coord.__dir__())
# print("===========================================================")
# print("===========================================================")
# cube_num = 4
# print("Data:", cubes[cube_num].standard_name, cubes[cube_num].data.shape, cubes[cube_num].data.shape, type(cubes[cube_num].data), cubes[cube_num].data)
# print("===========================================================")
# print(cubes[cube_num]._dim_coords_and_dims[0][0], type(cubes[cube_num]._dim_coords_and_dims[0][0]))
# print("-")
# print(cubes[cube_num]._dim_coords_and_dims[1][0], type(cubes[cube_num]._dim_coords_and_dims[1][0]))
# print("-")
# print(cubes[cube_num]._dim_coords_and_dims[0][1], type(cubes[cube_num]._dim_coords_and_dims[0][1]))
# print("-")
# print(cubes[cube_num]._dim_coords_and_dims[1][1], type(cubes[cube_num]._dim_coords_and_dims[1][1]))
# print("===========================================================")
# print(cubes[cube_num].coord('latitude').points, type(cubes[cube_num].coord('latitude').points))
# print(cubes[cube_num].coord('longitude').points - 360, type(cubes[cube_num].coord('longitude').points))
# print(cubes[cube_num].coord('model_level_number').points, type(cubes[cube_num].coord('model_level_number').points))
# print(cubes[cube_num].coord('time').points, type(cubes[cube_num].coord('time').points))
# print(cubes[cube_num].coord('forecast_reference_time'), cubes[cube_num].coord('forecast_reference_time').points)
# print("===========================================================")
# print("===========================================================")