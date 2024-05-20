import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta
from shapely.geometry import Point
import numpy as np
import iris.quickplot as qplt
import iris_grib
import re
from cartopy import crs
import sys

sys.path.append(os.getcwd())
from data_utils.extraction_funcs import ExtractGRIBIris
from misc.misc_utils import FindDate
#====================================================================
def extract_pattern(s):
    # This regex pattern looks for two numbers followed by anything, repeated three times
    pattern = r'(\d),(\d),.*?,(\d),(\d),.*?,(\d),(\d)'
    match = re.search(pattern, s)
    if match:
        return match.group(0)  # Returns the entire match
    else:
        return "No match found"
#====================================================================
''' Get data '''

base_path = '/Users/joshuamiller/Documents/Lancaster/Data/TempTest-ERA5'
month = '01'

file_list = []
for file in os.listdir(base_path):
    if file.endswith('.grib'):
        if (('2018-'+month in file) and ('137' in file)):
            file_list.append(file)

file_list.sort()

dict_ = ExtractGRIBIris(os.path.join(base_path, file_list[0]),
                        'all',
                        print_keys=False,
                        print_sum=False,
                        use_dask_array=False,
                        essential_var_names=['time', 'latitude', 'longitude'])
#print(dict_)

#====================================================================
''' Get plot ready '''
fig, ax = plt.subplots(4, 6, figsize=(14, 7))

cmap = 'bwr'
norm = Normalize(vmin=250, vmax=350)
world = gpd.read_file("/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")
# Generate some data and plot it in each subplot
for i in range(4):
    for j in range(6):

        dict_ = ExtractGRIBIris(os.path.join(base_path, file_list[6*i+j]),
                        'all',
                        print_keys=False,
                        print_sum=False,
                        use_dask_array=False,
                        essential_var_names=['time', 'latitude', 'longitude'])
        
        temp = np.squeeze(dict_['air_temperature'])
        time = str(dict_['forecast_reference_time'])
        lat = dict_['latitude']
        lon = dict_['longitude'] - 360
        lat_tiled = np.tile(lat, (np.shape(lon)[0], 1)).T
        lon_tiled = np.tile(lon, (np.shape(lat)[0], 1))
        
        scat = ax[i][j].scatter(x=lon_tiled, y=lat_tiled, c=temp, norm=norm, cmap='bwr')
        world.plot(ax=ax[i][j], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map

        ax[i][j].set_xlim(-21, 61)
        ax[i][j].set_ylim(-21, 21)
        ax[i][j].set_title(month+' '+time[10:])

# Create an axis on the left side for the colorbar
fig.subplots_adjust(left=0.1, right=0.8)
cbar_ax = fig.add_axes([0.81, 0.14, 0.02, 0.71])
cbar = fig.colorbar(scat, cax=cbar_ax)
cbar.set_label('Temperature (K)', rotation=270, labelpad=15)
#====================================================================
fig.savefig(os.path.join('/Users/joshuamiller/Documents/Lancaster/Figs', 'temp_'+str(month)+'.pdf'), 
            bbox_inches='tight', pad_inches=0)
plt.show()