import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
from shapely.geometry import Point
import netCDF4 as nc
import numpy as np

from preprocessing_funcs import Scale
from extraction_funcs import Extract_netCDF4
#====================================================================
''' Get data '''

dict_ = Extract_netCDF4("/Users/joshuamiller/Documents/Lancaster/Data/L2_O3_TCL/S5P_RPRO_L2__O3_TCL_20220713T111053_20220719T115620_24645_03_020401_20230329T142441.nc",
                        ['latitude_ccd', 'longitude_ccd', 'time', 'ozone_tropospheric_vertical_column'],
                        groups=['PRODUCT'],
                        print_sum=True)
#====================================================================
''' Make subplot '''
fig, ax = plt.subplots(figsize=(8, 6))

#ax.set_xlim(min(dict_['longitude_ccd']) - .1, max(dict_['longitude_ccd']) + .1)
#ax.set_ylim(min(dict_['latitude_ccd']) - .1, max(dict_['latitude_ccd']) + .1)
#====================================================================
''' Plot world map '''
world = gpd.read_file("/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")

world.plot(ax=ax, color='white', edgecolor='black', linewidth=0.1, alpha=1, legend=True) # GOOD lots the map
#====================================================================
''' Plot ozone '''
date = 0
ozone = dict_['ozone_tropospheric_vertical_column'][date, :, :]

# - - - - - - - - - Get points for the ozone plot - - - - - - - - - - -
lat = np.tile(dict_['latitude_ccd'], (np.shape(dict_['longitude_ccd'])[0], 1)).T
lon = np.tile(dict_['longitude_ccd'], (np.shape(dict_['latitude_ccd'])[0], 1))

points = [Point(x,y) for x,y in zip(lon.ravel(), lat.ravel())]
points_gdf = gpd.GeoDataFrame(geometry=points)

print('lat:', np.shape(lat),
      ', lon:', np.shape(lon),
      ', ozone:', np.shape(ozone),
      ', points:', np.shape(points))

del(lat)
del(lon)

ozone_gdf = gpd.GeoDataFrame(geometry=points).assign(data=ozone.ravel())

# - - - - - - - - - - - Make colorbar for ozone - - - - - - - - - - -
ozone_norm = Normalize(vmin=0, vmax=max(dict_['ozone_tropospheric_vertical_column'].ravel()))
ozone_cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'red'], N=200) # Higher N=more smooth

# - - - - - - - - - - - - - Plot ozone data - - - - - - - - - - - - -
ozone_gdf.plot(ax=ax, column='data', cmap=ozone_cmap, norm=ozone_norm, markersize=5, alpha=1, legend=True)
#====================================================================
plt.title(str(dict_['time']))

#====================================================================
plt.show()