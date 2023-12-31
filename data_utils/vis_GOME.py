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
from extraction_funcs import ExtractHDF5
from misc.misc_utils import DownSample
#====================================================================
''' Parse command line for file names '''
parser = argparse.ArgumentParser(description='Get file locations')
parser.add_argument('--data', nargs=1, type=str, help='location of the GOME data file',
                    default="/Users/joshuamiller/Documents/Lancaster/Data/Gome/S-O3M_GOME_OHP_02_M01_20210601011158Z_20210601020258Z_N_O_20210601082140Z.hdf5")
parser.add_argument('--map', nargs=1, type=str, help='location of the world map shape file',
                    default="/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")
args = parser.parse_args()
#====================================================================
''' Get data '''
# ['LatitudeCenter', 'LongitudeCenter', 'Time', 'IntegratedVerticalProfile']
dict_ = ExtractHDF5(args.data,
                    ['LatitudeCenter', 'LongitudeCenter', 'Time', 'IntegratedVerticalProfile'],
                    groups='all',
                    print_sum=True)
#====================================================================
downsample_rate = 1
#====================================================================
''' Make subplot '''
fig, ax = plt.subplots(figsize=(8, 6))
#====================================================================
''' Plot world map '''
world = gpd.read_file(args.map)

world.plot(ax=ax, color='white', edgecolor='black', linewidth=0.1, alpha=1, legend=True) # GOOD lots the map
#====================================================================
''' Plot ozone '''
date = 0
ozone = DownSample(dict_['IntegratedVerticalProfile'], 
                   downsample_rate=downsample_rate,
                   axis=0,
                   delete=True)

print('===== Loaded ozone data, shape:', np.shape(ozone))
# - - - - - - - - - Get points for the ozone plot - - - - - - - - - - -
points = [Point(x,y) for x,y in zip(dict_['LongitudeCenter'], dict_['LatitudeCenter'])]
print('===== Made points')

print('lat:', np.shape(dict_['LatitudeCenter']),
      ', lon:', np.shape(dict_['LongitudeCenter']),
      ', time:', np.shape(dict_['Time']),
      ', ozone:', np.shape(ozone),
      ', points:', np.shape(points))

ozone_gdf = gpd.GeoDataFrame(geometry=points).assign(data=ozone.ravel())

# - - - - - - - - - - - Make colorbar for ozone - - - - - - - - - - -
ozone_norm = Normalize(vmin=0, vmax=max(dict_['IntegratedVerticalProfile'].ravel()))
ozone_cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'red'], N=200) # Higher N=more smooth

# - - - - - - - - - - - - - Plot ozone data - - - - - - - - - - - - -
ozone_gdf.plot(ax=ax, column='data', cmap=ozone_cmap, norm=ozone_norm, markersize=1, alpha=1, legend=True)
#====================================================================
plt.title(str(dict_['Time']))

#====================================================================
plt.show()