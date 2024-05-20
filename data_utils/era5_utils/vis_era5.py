import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
from data_utils.extraction_funcs import ExtractGRIBIris, PrintSumGRIBIris, Extract_netCDF4
from misc.misc_utils import FindDate, Scale
#====================================================================
''' Get data '''

path1 = "/Users/joshuamiller/Documents/Lancaster/Data/Copernicus/adaptor.mars_constrained.external-1697707401.7930443-25417-7-468c83e5-7003-4cc1-b631-9d7c806b74c3.grib"
path3 = "/Users/joshuamiller/Documents/Lancaster/Data/Test-ERA5/ERA5-ml-temperature-subarea.nc"
path4 = "/Users/joshuamiller/Documents/Lancaster/Data/Test-ERA5/ERA5-ml-uwind-vwind-subarea_025025.nc"
path5 = "/Users/joshuamiller/Documents/Lancaster/Data/Test-ERA5/ERA5-ml-temperature-subarea_025025.grib"
path6 = "/Users/joshuamiller/Documents/Lancaster/Data/Test-ERA5/ERA5_p=131-132_l=1-137_2018-04-30.grib"

dict_ = ExtractGRIBIris(path6,
                        var_names='all',
                        sclice_over_var='model_level_number',
                        print_keys=True,
                        print_sum=True,
                        num_examples=1,
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
cubes = iris_grib.load_cubes(path6)
cubes = list(cubes)
cube_num = 0
#PrintSumGRIBIris(cubes, 2)

# Get the coordinate string
coord_str = str(cubes[cube_num].coord('forecast_reference_time'))

# Extract the date string after the colon
date_str = FindDate(coord_str, 'points')

# Create a figure with two subplots
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': crs.PlateCarree()})

# Plot the y-wind on the first subplot
qplt.contourf(cubes[cube_num], 20, axes=ax1)
ax1.coastlines()
ax1.set_title('Iris - '+ str(cubes[cube_num].standard_name)+' Level: '+str(cubes[cube_num].coord('model_level_number').points[0])+', date: '+date_str)

# Plot the x-wind on the second subplot
qplt.contourf(cubes[cube_num + 1], 20, axes=ax2)
ax2.coastlines()
ax2.set_title('Iris - '+ str(cubes[cube_num + 1].standard_name)+' Level: '+str(cubes[cube_num + 1].coord('model_level_number').points[0])+', date: '+date_str)

#====================================================================
#====================================================================
#====================================================================
#====================================================================
fig2, ((ax3, ax4), (ax5, ax6)) = plt.subplots(2, 2, figsize=(10, 5))
''' Plot world map '''
world = gpd.read_file("/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")
#====================================================================
''' Plot wind '''

date_idx = 0
level_idx = 0
x_wind = dict_['x_wind'][date_idx, level_idx, :, :]
y_wind = dict_['y_wind'][date_idx, level_idx, :, :]
# - - - - - - - - - Get points for the ozone plot - - - - - - - - - - -
lat = np.tile(dict_['latitude'], (np.shape(dict_['longitude'])[0], 1)).T
lon = np.tile(dict_['longitude']-360, (np.shape(dict_['latitude'])[0], 1))

points = [Point(x,y) for x,y in zip(lon.ravel(), lat.ravel())]

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
#====================================================================
#====================================================================
x_wind = ax5.scatter(x=lon, y=lat, c=x_wind, cmap='viridis', s=1)
y_wind = ax6.scatter(x=lon, y=lat, c=y_wind, cmap='viridis', s=1)
#====================================================================
divider = make_axes_locatable(ax5)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(x_wind, cax=cax, label='Velocity (m/s)')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=180)

divider = make_axes_locatable(ax6)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(y_wind, cax=cax, label='Velocity (m/s)')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=180)
#====================================================================
world.plot(ax=ax5, facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
world.plot(ax=ax6, facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
#====================================================================
ax5.set_title('Scatter - x_wind, level: ' + str(dict_['model_level_number'][level_idx]) + ', date: ' + dict_['forecast_reference_time'][date_idx])
ax6.set_title('Scatter - y_wind, level: ' + str(dict_['model_level_number'][level_idx]) + ', date: ' + dict_['forecast_reference_time'][date_idx])
#====================================================================
ax3.set_xlim(-21, 61)
ax4.set_xlim(-21, 61)
ax3.set_ylim(-21, 21)
ax4.set_ylim(-21, 21)
ax5.set_xlim(-21, 61)
ax6.set_xlim(-21, 61)
ax5.set_ylim(-21, 21)
ax6.set_ylim(-21, 21)
#====================================================================
#====================================================================
#====================================================================
#====================================================================
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
fig_, ax_ = plt.subplots(1,1,figsize=(7,7))

#path7 = "/Users/joshuamiller/Documents/Lancaster/Data/TempTest-ERA5/ERA5_p=temp_l=137_2018-04-29_0510TEST.grib"
path7 = "/Users/joshuamiller/Documents/Lancaster/Data/Temp-ERA5/ERA5_p=temp_l=137_2018-04-29_14:00:00.grib"

dict_ = ExtractGRIBIris(path7,
                        var_names='all',
                        sclice_over_var='model_level_number',
                        print_keys=True,
                        print_sum=True,
                        num_examples=1,
                        use_dask_array=False)

lat = np.tile(dict_['latitude'], (np.shape(dict_['longitude'])[0], 1)).T
lon = np.tile(dict_['longitude']-360, (np.shape(dict_['latitude'])[0], 1))
temp = np.squeeze(dict_['air_temperature'])

temp_plot = ax_.scatter(x=lon, y=lat, c=temp, s=1, cmap='bwr')
world.plot(ax=ax_, facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map

divider = make_axes_locatable(ax_)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(temp_plot, cax=cax, label='Temperaute (K)')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=180)
ax_.set_xlim(-21, 61)
ax_.set_ylim(-21, 21)

#fig_.savefig("/Users/joshuamiller/Documents/Lancaster/Figs/temp_1005.pdf", bbox_inches='tight', pad_inches=0)
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
#====================================================================
#====================================================================
#====================================================================
#====================================================================
path8        = '/Users/joshuamiller/Documents/Lancaster/Data/Temp-ERA5/ERA5_p=temp_l=137_2020-07-04_14:00:00.grib'
path8_big    = '/Users/joshuamiller/Documents/Lancaster/Data/Uwind-ERA5/ERA5_p=Uwind_l=135_2018-04-28_2022-12-31_14:00:00.grib'
path8_big_nc = '/Users/joshuamiller/Documents/Lancaster/Data/Whole_Area/Temp/ERA5_p=temp_l=137_2018-04-28_2022-12-31_14:00:00.nc'

fig8, ax8 = plt.subplots(1,3,figsize=(12, 5), subplot_kw={'projection': crs.PlateCarree()})
#====================================================================
cubes = iris_grib.load_cubes(path8)
cubes = list(cubes)
cube_num = 0

coord_str = str(cubes[cube_num].coord('forecast_reference_time'))
date_str = FindDate(coord_str, 'points')

qplt.contourf(cubes[cube_num], 20, axes=ax8[0])
ax8[0].coastlines()
ax8[0].set_title('Iris - '+ str(cubes[cube_num].standard_name)+' Level: '+str(cubes[cube_num].coord('model_level_number').points[0])+', date: '+date_str)
#====================================================================
var_name = 'x_wind'

dict_ = ExtractGRIBIris(path8_big,
                        var_names='all',
                        sclice_over_var='model_level_number',
                        print_keys=True,
                        print_sum=True,
                        num_examples=1,
                        use_dask_array=False)

date_idx1 = 798 # 3

lat = np.tile(dict_['latitude'], (np.shape(dict_['longitude'])[0], 1)).T
lon = np.tile(dict_['longitude']-360, (np.shape(dict_['latitude'])[0], 1))
level = dict_['model_level_number']
data = np.squeeze(dict_[var_name])

date1 = dict_['forecast_reference_time'][date_idx1]

plot1 = ax8[1].scatter(x=lon, y=lat, c=data[date_idx1, :, :], s=1, cmap='viridis')
world.plot(ax=ax8[1], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map

# divider = make_axes_locatable(ax8[1])
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cbar = plt.colorbar(v_wind_plot, cax=cax, label='Velocity (m/s)')
# cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=180)

ax8[1].set_title('Manual .grib\n'+var_name+', '+str(level)+', '+date1)
#====================================================================
print('-+-=-+-+-=-+-+-=-+-+-=-+-+-=-+-+-=-+-=-+')
print('++++++++++++++++++++++++++++++++++++++++')
print('-+-=-+-+-=-+-+-=-+-+-=-+-+-=-+-+-=-+-=-+')
var_name = 'air_temperature'
dict_nc = Extract_netCDF4(path8_big_nc,
                          var_names=[var_name, 'time', 'latitude', 'longitude', 'forecast_reference_time', 'model_level_number'],
                          groups='all',
                          print_sum=True)

lat_nc = np.tile(dict_nc['latitude'], (np.shape(dict_nc['longitude'])[0], 1)).T
lon_nc = np.tile(dict_nc['longitude'], (np.shape(dict_nc['latitude'])[0], 1))
level_nc = dict_nc['model_level_number']
data_nc = np.squeeze(dict_nc[var_name])

date_idx2 = 798

hours      = dict_nc['forecast_reference_time']
start_date = datetime(1970, 1, 1, 0, 0, 0)
new_dates  = [start_date + timedelta(hours=hour) for hour in hours]
date2 = new_dates[date_idx2]


plot2 = ax8[2].scatter(x=lon_nc, y=lat_nc, c=data_nc[date_idx2, :, :], s=1, cmap='viridis')
world.plot(ax=ax8[2], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map

ax8[2].set_title('Manual netCDF4\n'+var_name+', '+str(level_nc)+', '+date2.strftime('%Y-%m-%d %H:%M:%S'))
#====================================================================
ax8[0].set_xlim(-21, 61)
ax8[0].set_ylim(-21, 21)
ax8[1].set_xlim(-21, 61)
ax8[1].set_ylim(-21, 21)
ax8[2].set_xlim(-21, 61)
ax8[2].set_ylim(-21, 21)
























plt.show()


print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('lon=', np.shape(dict_['longitude']-360), dict_['longitude']-360)
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('lat=', np.shape(dict_['latitude']), dict_['latitude'])
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++')



















print("IRIS:", len(cubes), type(cubes[0])) 
print("===========================================================")
print("===========================================================")
print("vars(cubes[0]):", vars(cubes[0]))
print("===========================================================")
print("===========================================================")
print("cubes[0].__dir__:", cubes[0].__dir__())
print("===========================================================")
print("===========================================================")
print("cubes[0].__dict__:", cubes[0].__dict__)
print("===========================================================")
print("===========================================================")
print("cubes[0].data.__dir__():", cubes[0].units, cubes[0].data.__dir__())
print("===========================================================")
print("===========================================================")
print("cubes[0].coord.__dir__():", cubes[0].coord.__dir__())
print("===========================================================")
print("===========================================================")
# # cube_num = 4
# # print("Data:", cubes[cube_num].standard_name, cubes[cube_num].data.shape, cubes[cube_num].data.shape, type(cubes[cube_num].data), cubes[cube_num].data)
# # print("===========================================================")
# # print(cubes[cube_num]._dim_coords_and_dims[0][0], type(cubes[cube_num]._dim_coords_and_dims[0][0]))
# # print("-")
# # print(cubes[cube_num]._dim_coords_and_dims[1][0], type(cubes[cube_num]._dim_coords_and_dims[1][0]))
# # print("-")
# # print(cubes[cube_num]._dim_coords_and_dims[0][1], type(cubes[cube_num]._dim_coords_and_dims[0][1]))
# # print("-")
# # print(cubes[cube_num]._dim_coords_and_dims[1][1], type(cubes[cube_num]._dim_coords_and_dims[1][1]))
# # print("===========================================================")
# # print(cubes[cube_num].coord('latitude').points, type(cubes[cube_num].coord('latitude').points))
# # print(cubes[cube_num].coord('longitude').points - 360, type(cubes[cube_num].coord('longitude').points))
# # print(cubes[cube_num].coord('model_level_number').points, type(cubes[cube_num].coord('model_level_number').points))
# # print(cubes[cube_num].coord('time').points, type(cubes[cube_num].coord('time').points))
# # print(cubes[cube_num].coord('forecast_reference_time'), cubes[cube_num].coord('forecast_reference_time').points)
# # print("===========================================================")
# # print("===========================================================")