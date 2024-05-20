import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
import geopandas as gpd
import netCDF4 as nc
import numpy as np
import sys
from datetime import date, datetime, timedelta

sys.path.append(os.getcwd())
from data_utils.preprocessing_funcs import Scale
from data_utils.extraction_funcs import Extract_netCDF4
from misc.misc_utils import GetBoxCoords
#====================================================================
''' Folders '''
region      = 'North_Land'
variable    = ['Temp', 'temp', 'air_temperature']
#variable    = ['Uwind', 'Uwind', 'x_wind']
#variable    = ['Vwind', 'Vwind', 'y_wind']
save_folder = "/Users/joshuamiller/Documents/Lancaster/Data/"+region+"/"+variable[0]
data_folder = "/Users/joshuamiller/Documents/Lancaster/Data/Whole_Area/"+variable[0]
file        = "ERA5_p="+variable[1]+"_l=137_2018-04-28_2022-12-31_Whole_Area.nc"
#====================================================================
dict_ = GetBoxCoords("data_utils/data_utils_config.yml")
min_lat = dict_[region][3]
max_lat = dict_[region][1]
min_lon = dict_[region][0]
max_lon = dict_[region][2]
print(min_lat, max_lat, min_lon, max_lon)
#====================================================================
''' Search for nc files '''

dict_ = Extract_netCDF4(os.path.join(data_folder, file),
                        ['latitude', 'longitude', 'time', 'model_level_number', variable[2]],
                        groups='all',
                        print_sum=True)

dataset_orig = nc.Dataset(os.path.join(data_folder, file))

lat   = dict_['latitude']
lon   = dict_['longitude'] - 360
hours = dict_['time']
level = dict_['model_level_number']
data  = dict_[variable[2]]

dates = [datetime(1970, 1, 1)+timedelta(hours=day) for day in hours]
#--------------------------------------------------------
valid_lat_idx = []
for lat_idx in range(np.shape(lat)[0]):
    if (min_lat <= lat[lat_idx] and lat[lat_idx] <= max_lat):
        valid_lat_idx.append(int(lat_idx))

valid_lon_idx = []
for lon_idx in range(np.shape(lon)[0]):
    if (min_lon <= lon[lon_idx] and lon[lon_idx] <= max_lon):
        valid_lon_idx.append(int(lon_idx))

valid_time_idx = []
for time_idx in range(np.shape(dates)[0]):
    if (datetime(2018, 4, 30) <= dates[time_idx] and dates[time_idx] <= datetime(2022, 8, 1)):
        valid_time_idx.append(int(time_idx))
#--------------------------------------------------------
filtered_data = np.ones((len(valid_time_idx), len(valid_lat_idx), len(valid_lon_idx)), float) * -999

print("l=", dict_['model_level_number'], ', time=', np.shape(valid_time_idx), ', lat=', np.shape(valid_lat_idx), ', lon=', np.shape(valid_lon_idx))

for i in range(len(valid_time_idx)):
    for j in range(len(valid_lat_idx)):
        for k in range(len(valid_lon_idx)):
            filtered_data[i, j, k] = dict_[variable[2]][valid_time_idx[i], valid_lat_idx[j], valid_lon_idx[k]]
#--------------------------------------------------------

#--------------------------------------------------------

#--------------------------------------------------------
start_date = datetime.strftime(dates[valid_time_idx[0]], '%Y-%m-%d')
end_date   = datetime.strftime(dates[valid_time_idx[-1]], '%Y-%m-%d')

filename = "ERA5_p="+variable[1]+"_l="+str(dict_['model_level_number'])+'_'+start_date+'_'+end_date+'_'+region+'.nc'
print('filename=', filename)

units = ''
if variable[0] == 'Temp':
    units = 'degrees_Kelvin'
elif ((variable[0] == 'Uwind') or (variable[0] == 'Vwind')):
    units = 'm s-1'


dataset = nc.Dataset(os.path.join(save_folder, filename), 'w')

lat_dim   = dataset.createDimension('lat', len(valid_lat_idx))
lon_dim   = dataset.createDimension('lon', len(valid_lon_idx))
time_dim  = dataset.createDimension('time', len(valid_time_idx))
level_dim = dataset.createDimension('model_level_number', 1)

lat_var   = dataset.createVariable('lat', 'f4', ('lat',))
lon_var   = dataset.createVariable('lon', 'f4', ('lon',))
time_var  = dataset.createVariable('time', np.float64, ('time',))
data_var  = dataset.createVariable(variable[2], 'f4', ('time', 'lat', 'lon',))
level_var = dataset.createVariable('model_level_number', 'i4', ('model_level_number',))

lat_var.units  = 'degrees_north'
lon_var.units  = 'degrees_east'
time_var.units = 'hours since 1970-01-01 00:00:00'
data_var.units = units

lat_var[:]        = [lat[idx] for idx in valid_lat_idx]
lon_var[:]        = [lon[idx] for idx in valid_lon_idx]
time_var[:]       = [hours[idx] for idx in valid_time_idx]
data_var[:, :, :] = filtered_data
level_var[:]      = dict_['model_level_number']

dataset.title = variable[2]+', model level: '+ str(dict_['model_level_number'])+', start date: '+start_date+', end date: '+end_date
dataset.institution = 'Lancaster University'
dataset.source = "Josh Miller - created by: spat_filter_era5.py\nsource file: "+file+"\nData originally from European Centre for Medium Range Weather Forecasts ERA5"
dataset.description = "Data from ERA5 has been spatially filtered to lat: ["+str(min_lat)+", "+str(max_lat)+"], lon: ["+str(min_lon)+", "+str(max_lon)+"]."
dataset.history = 'Created on '+str(date.today())
dataset.setncattr('crs', 'EPSG:4326')

dataset.close()
dataset_orig.close()