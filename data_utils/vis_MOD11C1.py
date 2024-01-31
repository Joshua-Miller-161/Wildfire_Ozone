import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
from shapely.geometry import Point
import numpy as np
import argparse
import fiona
import sys
#====================================================================
sys.path.append('/Users/joshuamiller/Documents/Python Files/Wildfire_Ozone')
#====================================================================
print(fiona.__version__)
from preprocessing_funcs import Scale
from extraction_funcs import ExtractHDF5, Extract_netCDF4
from misc.misc_utils import DownSample
#====================================================================
''' Get data '''
# ['LatitudeCenter', 'LongitudeCenter', 'Time', 'IntegratedVerticalProfile']
dict_ = Extract_netCDF4("/Users/joshuamiller/Documents/Lancaster/Data/20190430120000-C3S-L4_GHRSST-SSTdepth-OSTIA-GLOB_ICDR2.1-v02.0-fv01.0.nc",
                        ['lat', 'lon', 'time', 'analysed_sst'],
                         groups='all',
                         print_sum=True)
for lol in dict_.keys():
    print(np.shape(dict_[lol]))

print("______")
print(min(dict_['analysed_sst']),max(dict_['analysed_sst']), type(dict_['analysed_sst']))
print("______")
#====================================================================
downsample_rate = 10

lat = DownSample(dict_['lat'], downsample_rate=downsample_rate, axis=0, delete=True)
lon = DownSample(dict_['lon'], downsample_rate=downsample_rate, axis=0, delete=True)
temp = DownSample(DownSample(dict_['analysed_sst'], downsample_rate=downsample_rate, axis=1, delete=True), downsample_rate=downsample_rate, axis=2, delete=True)
time = dict_['time']

del(dict_)
print(np.shape(lat), np.shape(lon), np.shape(temp))
#====================================================================
''' Make subplot '''
fig, ax = plt.subplots(figsize=(8, 6))
#====================================================================
''' Plot world map '''
world = gpd.read_file("/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")

world.plot(ax=ax, color='white', edgecolor='black', linewidth=0.1, alpha=1, legend=True) # GOOD lots the map
#====================================================================
''' Plot ozone '''
lat_ = np.tile(lat, (np.shape(lon)[0], 1)).T
lon_ = np.tile(lon, (np.shape(lat)[0], 1))

del(lat)
del(lon)
# - - - - - - - - - Get points for the ozone plot - - - - - - - - - - -
points = [Point(x,y) for x,y in zip(lat_.ravel(), lon_.ravel())]
print('===== Made points', np.shape(points))


ozone_gdf = gpd.GeoDataFrame(geometry=points).assign(data=temp.ravel())

# - - - - - - - - - - - Make colorbar for ozone - - - - - - - - - - -
ozone_norm = Normalize(vmin=0, vmax=max(temp.ravel()))
ozone_cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'red'], N=200) # Higher N=more smooth

# - - - - - - - - - - - - - Plot ozone data - - - - - - - - - - - - -
ozone_gdf.plot(ax=ax, column='data', cmap=ozone_cmap, norm=ozone_norm, markersize=1, alpha=1, legend=True)
#====================================================================
plt.title(str(time))

#====================================================================
plt.show()