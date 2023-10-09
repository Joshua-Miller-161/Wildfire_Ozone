import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
from shapely.geometry import Point
import numpy as np
import argparse
import fiona

print(fiona.__version__)
from preprocessing_funcs import Scale
from extraction_funcs import ExtractHDF5
#from misc.misc_utils import DownSample
#====================================================================
def DownSample(data, downsample_rate, axis, delete=False):
    '''
    Made by ChatGPT - Extracts data points separated by skip along the given axis

    Returns downsampled data.
    
    - data (ndarray) - the data
    - skip (int) - the number of elements that are skiped when downsampling
    - axis (int) - the axis on which to downsample
    - delete (bool, optional) - whether or not to delete the original data in order to save memory
    '''
    slices       = [slice(None)] * data.ndim
    slices[axis] = slice(None, None, downsample_rate)
    new_data     = data[tuple(slices)]
    
    print('Orig. shape :', np.shape(data), "----> new shape :", np.shape(new_data))

    if delete:
        del(data)

    return new_data
#====================================================================
''' Parse command line for file names '''
parser = argparse.ArgumentParser(description='Get file locations')
parser.add_argument('data', type=str, help='location of the GOME data file')
parser.add_argument('map', type=str, help='location of the world map shape file')
args = parser.parse_args()

data_path = args.data
map_path = args.map

# "/Users/joshuamiller/Documents/Lancaster/Data/Gome/S-O3M_GOME_OHP_02_M01_20210601011158Z_20210601020258Z_N_O_20210601082140Z.hdf5"

#====================================================================
''' Get data '''
dict_ = ExtractHDF5(data_path,
                    ['LatitudeCenter', 'LongitudeCenter', 'Time', 'IntegratedVerticalProfile'],
                    groups=['DATA', 'GEOLOCATION'],
                    print_sum=True,
                    to_numpy=True)
#====================================================================
downsample_rate = 2
#====================================================================
''' Make subplot '''
fig, ax = plt.subplots(figsize=(8, 6))
#====================================================================
''' Plot world map '''

# "/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp"

#world = gpd.read_file(map_path)

#world.plot(ax=ax, color='white', edgecolor='black', linewidth=0.1, alpha=1, legend=True) # GOOD lots the map
#====================================================================
''' Plot ozone '''
date = 0
ozone = DownSample(dict_['IntegratedVerticalProfile'], 
                   downsample_rate=downsample_rate,
                   axis=0,
                   delete=True)

print('===== Loaded ozone data, shape:', np.shape(ozone))
# - - - - - - - - - Get points for the ozone plot - - - - - - - - - - -
lat = DownSample(dict_['LatitudeCenter'], 
                   downsample_rate=downsample_rate,
                   axis=0,
                   delete=True)
lon = DownSample(dict_['LongitudeCenter'], 
                   downsample_rate=downsample_rate,
                   axis=0,
                   delete=True)

lat_tiled = np.tile(lat, (np.shape(lon)[0], 1)).T
print('===== Tiled latitude, shape:', np.shape(lat_tiled))
lon_tiled = np.tile(lon, (np.shape(lat)[0], 1))
print('===== Tiled longitude, shape:', np.shape(lon_tiled))

points = [Point(x,y) for x,y in zip(lon, lat)]
print('===== Made points')

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