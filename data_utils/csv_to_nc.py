import sys
sys.dont_write_bytecode
import os
os.environ['USE_PYGEOS'] = '0'
import netCDF4 as nc
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(os.getcwd())
from misc.misc_utils import GetDateInStr
from data_utils.extract_netCDF4 import Extract_netCDF4
from vis.plotting_utils import PlotBoxes
#====================================================================
region = 'Whole_Area'
folder_path = "/Users/joshuamiller/Documents/Lancaster/Data/Kriged_L2_O3_TCL_v2"
dest_file = "/Users/joshuamiller/Documents/Lancaster/Data/"+region+"/Ozone/S5P_RPRO_L2__O3_TCL_2018-04-30_2022-07-31_"+region+".nc"

file1 = '/Users/joshuamiller/Documents/Lancaster/Data/L2_O3_TCL/S5P_RPRO_L2__O3_TCL_20200725T121326_20200731T125843_14459_03_020401_20230329T125441/S5P_RPRO_L2__O3_TCL_20200725T121326_20200731T125843_14459_03_020401_20230329T125441.nc'
file2 = dest_file
var_to_plot = 'ozone_tropospheric_vertical_column'
#====================================================================
# Function to read and concatenate .csv files
def read_and_concatenate_csv(files):
    data_frames = []
    for file in files:
        date = GetDateInStr(file)
        df = pd.read_csv(file)
        df['time'] = date
        data_frames.append(df)
    concatenated_df = pd.concat(data_frames)
    return concatenated_df
#--------------------------------------------------------------------
# Function to read and concatenate .nc files
def read_and_concatenate_netCDF4(files, var_names):
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ''' Get a reference to initialize stuff '''

    dict_ = Extract_netCDF4(files[0],
                            ['lat', 'lon']+var_names,
                            groups='all',
                            print_sum=False)
    lat = dict_['lat']
    lon = dict_['lon']
    dates_big = ['init'] * len(files)

    data_dict = {}
    for i in range(len(var_names)):
        data_dict[var_names[i]] = np.ones((len(dates_big), np.shape(lon)[0], np.shape(lat)[0]), float) * -999
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ''' Go through all files and extract data '''
    for i in range(len(files)):
        dict_ = Extract_netCDF4(files[i],
                                var_names,
                                groups='all',
                                print_sum=False)
        for var_name in var_names:
            data_dict[var_name][i, :, :] = np.squeeze(dict_[var_name])

        dataset_orig = nc.Dataset(files[i])
        dates_big[i] = dataset_orig.time_coverage_start[:10]
        dataset_orig.close()
        #print("lat=", np.shape(lat), ", lon=", np.shape(lon), ", O3=", np.shape(ozone), dates_big[i])

    return lat, lon, dates_big, data_dict
#--------------------------------------------------------------------
def create_netCDF(output_filename, 
                  lat, lon, time, 
                  data_dict, var_names, var_units,
                  path_to_orig=None, message='', crs='EPSG:4326'):
    dataset = nc.Dataset(output_filename, 'w', format='NETCDF4')
    # Create dimensions
    time_dim = dataset.createDimension('start_date', np.shape(time)[0])  # None for unlimited dimension
    lat_dim = dataset.createDimension('lat', np.shape(lat)[0])
    lon_dim = dataset.createDimension('lon', np.shape(lon)[0])
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Create variables
    date_ = dataset.createVariable('start_date', np.float64, ('start_date',))
    lat_  = dataset.createVariable('lat', np.float32, ('lat',))
    lon_  = dataset.createVariable('lon', np.float32, ('lon',))

    data_vars = []
    for var in var_names:
        data_vars.append(dataset.createVariable(var, np.float32, ('start_date', 'lat', 'lon',)))
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Assign data to variables
    date_datetimes = [datetime.strptime(date, '%Y-%m-%d') for date in time]
    date_[:]       = nc.date2num(date_datetimes, units='days since 1970-01-01 00:00:00', calendar='gregorian')
    
    lat_[:] = lat
    lon_[:] = lon

    # Assuming 'data1' and 'data2' are 2D arrays with shape (time, lat*lon)
    for i in range(len(var_names)):
        data_var          = data_vars[i]
        data_var.units    = var_units[i]

        data = data_dict[var_names[i]]
        print("var_names[", i, "]=", var_names[i], ", data=", data)
        if (len(np.shape(data)) > 3):
            data_var[:, :, :] = data
        elif (len(np.shape(data)) == 3):
            data_var[:, :, :] = data
        elif (len(np.shape(data)) == 2):
            data_var[:, :, :] = data.reshape(-1, np.shape(lat)[0], np.shape(lon)[0])
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Add attributes to variables
    date_.units    = 'Days since 1970-01-01 00:00:00'
    date_.calendar = 'gregorian'
    lat_.units     = 'degrees_north'
    lon_.units     = 'degrees_east'

    dataset.Conventions = 'CF-1.8'
    dataset.title       = 'Tropospheric ozone column. Start date: '+dates[0]+' end date: '+dates[-1]
    dataset.institution = 'Lancaster University'
    dataset.history     = 'Created on '+str(date.today())
    dataset.description = message
    dataset.source      = "Joshua Miller - created by: csv_to_nc.py\nData originally from Sentinel-5 precursor/TROPOMI Level 2 Product O3 Tropospheric Column (L2__O3_TCL)"
    dataset.setncattr('crs', crs)

    if not (path_to_orig==None):
        dataset_orig = nc.Dataset(path_to_orig)
        dataset.setncattr('geospatial_vertical_range_top_troposphere', dataset_orig.geospatial_vertical_range_top_troposphere)
        dataset.setncattr('geospatial_vertical_range_bottom_stratosphere', dataset_orig.geospatial_vertical_range_bottom_stratosphere)
        dataset.setncattr('processor_version', dataset_orig.processor_version)
        dataset.setncattr('product_version', dataset_orig.product_version)
        dataset.setncattr('algorithm_version', dataset_orig.algorithm_version)
        dataset_orig.close()

    dataset.close()
#====================================================================
files = []
for file in os.listdir(folder_path):
    if (file.endswith('.csv') or file.endswith('.nc')):
        files.append(os.path.join(folder_path, file))
files.sort()

new_files = []
for file in files:
    date = GetDateInStr(file)
    date = datetime.strptime(date, '%Y%m%d')

    if ((datetime(2018, 4, 30) <= date) and (date <= datetime(2022, 7, 31))):
        new_files.append(file)

#print(new_files)
#--------------------------------------------------------------------
# concatenated_df = read_and_concatenate_csv(new_files)

lat, lon, dates, data_dict = read_and_concatenate_netCDF4(new_files, [var_to_plot, 'qa_value', 'krig_mask'])
#print(data_dict)
# print("lat", np.shape(lat), ", lon", np.shape(lon), ", dates", np.shape(dates))
# for key in data_dict.keys():
#     print(key, np.shape(data_dict[key]))

create_netCDF(dest_file,
              lat, lon, dates,
              data_dict, [var_to_plot, 'qa_value', 'krig_mask'], ['mol m-2', 'percent', '1: was kriged; 0: not kriged, i.e. original data'],
              path_to_orig=file1,
              message="'start_date' refers to the first day in the 5-day period over which the tropospheric column is averaged.")
#====================================================================
#====================================================================
#====================================================================
#====================================================================
world = gpd.read_file("/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
fig.subplots_adjust(wspace=.3)
#--------------------------------------------------------------------
# df = pd.read_csv(file1)
# lon_tiled = df.loc[:, 'lon']
# lat_tiled = df.loc[:, 'lat']
# data      = df.loc[:, var_to_plot]

dict_ = Extract_netCDF4(file1,
                        var_names=['longitude_ccd', 'latitude_ccd', 'dates_for_tropospheric_column', var_to_plot],
                        groups='all')
lon  = dict_['longitude_ccd']
lat  = dict_['latitude_ccd']
data = np.squeeze(dict_[var_to_plot])
lat_tiled = np.tile(lat, (np.shape(lon)[0], 1)).T
lon_tiled = np.tile(lon, (np.shape(lat)[0], 1))

min_ = min(data.ravel())
max_ = max(data.ravel())
if (min_ <= 0):
    min_ = 10**-1
#norm = LogNorm(vmin=min_, vmax=max_)
norm = Normalize(vmin=min_, vmax=max_)
cmap = LinearSegmentedColormap.from_list('custom', ['blue',
                                                    'cornflowerblue',
                                                    'powderblue',
                                                    'pink',
                                                    'palevioletred',
                                                    'red'], N=200) # Higher N=more smooth

scat = ax[0].scatter(x=lon_tiled, y=lat_tiled, c=data, s=2, cmap=cmap, norm=norm)
world.plot(ax=ax[0], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
PlotBoxes("data_utils/data_utils_config.yml", ax[0])

divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(scat, cax=cax, label='')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=180)

ax[0].set_title(GetDateInStr(file1))
#--------------------------------------------------------------------
dict_ = Extract_netCDF4(file2,
                        var_names=['lat', 'lon', 'start_date', var_to_plot],
                        groups='all',
                        print_sum=True)
date_idx = 817
lon = dict_['lon']
lat = dict_['lat']
dates = dict_['start_date']
data = dict_[var_to_plot][date_idx, :, :]

lat_tiled = np.tile(lat, (np.shape(lon)[0], 1)).T
lon_tiled = np.tile(lon, (np.shape(lat)[0], 1))

scat_ = ax[1].scatter(x=lon_tiled, y=lat_tiled, c=data, s=2, cmap=cmap, norm=norm)
world.plot(ax=ax[1], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
PlotBoxes("data_utils/data_utils_config.yml", ax[1])

curr_date = datetime(1970, 1, 1) + timedelta(days=dates[date_idx])
ax[1].set_title(curr_date.strftime('%Y-%m-%d %H:%M:%S'))
#--------------------------------------------------------------------
ax[0].set_xlim(-21, 61)
ax[0].set_ylim(-21, 21)
ax[1].set_xlim(-21, 61)
ax[1].set_ylim(-21, 21)
plt.show()