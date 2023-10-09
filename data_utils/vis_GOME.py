import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
from shapely.geometry import Point
import netCDF4 as nc
import numpy as np
import xarray as xr

from preprocessing_funcs import Scale
from extraction_funcs import ExtractHDF5
#====================================================================
''' Get data '''
dict_ = ExtractHDF5("/Users/joshuamiller/Documents/Lancaster/Data/Gome/S-O3M_GOME_OHP_02_M01_20210601011158Z_20210601020258Z_N_O_20210601082140Z.hdf5",
                    ['LatitudeCenter', 'LongitudeCenter', 'Time', 'IntegratedVerticalProfile'],
                    groups=['DATA', 'GEOLOCATION'],
                    print_sum=True,
                    to_numpy=True)
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
ozone = dict_['IntegratedVerticalProfile']

# - - - - - - - - - Get points for the ozone plot - - - - - - - - - - -
lat = np.tile(dict_['LatitudeCenter'], (np.shape(dict_['LongitudeCenter'])[0], 1)).T
lon = np.tile(dict_['LongitudeCenter'], (np.shape(dict_['LatitudeCenter'])[0], 1))

points = [Point(x,y) for x,y in zip(lon.ravel(), lat.ravel())]
points_gdf = gpd.GeoDataFrame(geometry=points)

print('lat:', np.shape(lat),
      ', lon:', np.shape(lon),
      ', ozone:', np.shape(ozone),
      ', points:', np.shape(points))

ozone_gdf = gpd.GeoDataFrame(geometry=points).assign(data=ozone.ravel())

# - - - - - - - - - - - Make colorbar for ozone - - - - - - - - - - -
ozone_norm = Normalize(vmin=0, vmax=max(dict_['IntegratedVerticalProfile'].ravel()))
ozone_cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'red'], N=200) # Higher N=more smooth

# - - - - - - - - - - - - - Plot ozone data - - - - - - - - - - - - -
ozone_gdf.plot(ax=ax, column='data', cmap=ozone_cmap, norm=ozone_norm, markersize=5, alpha=1, legend=True)
#====================================================================
plt.title(str(dict_['time']))

#====================================================================
plt.show()