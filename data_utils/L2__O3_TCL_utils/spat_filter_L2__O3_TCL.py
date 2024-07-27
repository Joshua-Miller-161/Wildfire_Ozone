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
from data_utils.extract_netCDF4 import Extract_netCDF4
from misc.misc_utils import GetBoxCoords
#====================================================================
''' Folders '''
region = 'North_Land'
search_folder = "/Users/joshuamiller/Documents/Lancaster/Data/L2_O3_TCL"
save_folder   = "/Users/joshuamiller/Documents/Lancaster/Data/"+region+"/Ozone"

data_folder = "/Users/joshuamiller/Documents/Lancaster/Data/Whole_Area/Ozone"
file = "S5P_RPRO_L2__O3_TCL_2018-04-30_2022-07-31_Whole_Area.nc"
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
                        ['lat', 'lon', 'start_date', 'ozone_tropospheric_vertical_column', 'qa_value', 'krig_mask'],
                        groups='all',
                        print_sum=False)

dataset_orig = nc.Dataset(os.path.join(data_folder, file))

dates = dict_['start_date']
dates = [datetime.strftime(datetime(1970, 1, 1)+timedelta(days=day), '%Y-%m-%d') for day in dates]
#--------------------------------------------------------
valid_lat_idx = []
for lat_idx in range(np.shape(dict_['lat'])[0]):
    if (min_lat <= dict_['lat'][lat_idx] and dict_['lat'][lat_idx] <= max_lat):
        valid_lat_idx.append(int(lat_idx))

valid_lon_idx = []
for lon_idx in range(np.shape(dict_['lon'])[0]):
    if (min_lon <= dict_['lon'][lon_idx] and dict_['lon'][lon_idx] <= max_lon):
        valid_lon_idx.append(int(lon_idx))
#--------------------------------------------------------
filtered_O3 = np.ones((len(dates), len(valid_lat_idx), len(valid_lon_idx)), float) * -999

for i in range(len(valid_lat_idx)):
    for j in range(len(valid_lon_idx)):
        filtered_O3[:, i, j] = dict_['ozone_tropospheric_vertical_column'][:, valid_lat_idx[i], valid_lon_idx[j]]
#--------------------------------------------------------
filtered_qa = np.empty((len(dates), len(valid_lat_idx), len(valid_lon_idx)), float) * -999

for i in range(len(valid_lat_idx)):
    for j in range(len(valid_lon_idx)):
        filtered_qa[:, i, j] = dict_['qa_value'][:, valid_lat_idx[i], valid_lon_idx[j]]
#--------------------------------------------------------
filtered_krig = np.empty((len(dates), len(valid_lat_idx), len(valid_lon_idx)), float) * -999

for i in range(len(valid_lat_idx)):
    for j in range(len(valid_lon_idx)):
        filtered_krig[:, i, j] = dict_['krig_mask'][:, valid_lat_idx[i], valid_lon_idx[j]]
#--------------------------------------------------------
filename = file[:19]+'_'+dates[0]+'_'+dates[-1]+'_'+region+'.nc'

dataset = nc.Dataset(os.path.join(save_folder, filename), 'w')

lat_dim = dataset.createDimension('lat', len(valid_lat_idx))
lon_dim = dataset.createDimension('lon', len(valid_lon_idx))
time_dim = dataset.createDimension('start_date', len(dict_['start_date']))

lat_var  = dataset.createVariable('lat', 'f4', ('lat',))
lon_var  = dataset.createVariable('lon', 'f4', ('lon',))
date_var = dataset.createVariable('start_date', np.float64, ('start_date',))
O3_var   = dataset.createVariable('ozone_tropospheric_vertical_column', 'f4', ('start_date', 'lat', 'lon',))
qa_var   = dataset.createVariable('qa_value', 'f4', ('start_date', 'lat', 'lon',))
krig_var = dataset.createVariable('krig_mask', 'f4', ('start_date', 'lat', 'lon',))

lat_var.units  = 'degrees_north'
lon_var.units  = 'degrees_east'
date_var.units = 'days since 1970-01-01 00:00:00'
O3_var.units   = 'mol m-2'
qa_var.units   = '%'
krig_var.units = '1: was kriged; 0: not kriged, i.e. original data'

lat_var[:]     = [dict_['lat'][idx] for idx in valid_lat_idx]
lon_var[:]     = [dict_['lon'][idx] for idx in valid_lon_idx]
date_var[:]    = dict_['start_date']
O3_var[:, :]   = filtered_O3
qa_var[:, :]   = filtered_qa
krig_var[:, :] = filtered_krig

dataset.Conventions = 'CF-1.8'
dataset.title = 'Tropospheric ozone column. Start date: '+dates[0]+' end date: '+dates[-1]
dataset.institution = 'Lancaster University'
dataset.source = "Josh Miller - created by: spat_filter_L2__O3_TCL.py\nsource file: "+file+"\nData originally from Sentinel-5 precursor/TROPOMI Level 2 Product O3 Tropospheric Column (L2__O3_TCL)"
dataset.description = "Kriged data from L2__O3_TCL has been spatially filtered to lat: ["+str(min_lat)+", "+str(max_lat)+"], lon: ["+str(min_lon)+", "+str(max_lon)+"]."
dataset.history = 'Created on '+str(date.today())
dataset.setncattr('crs', 'EPSG:4326')
dataset.setncattr('geospatial_vertical_range_top_troposphere', dataset_orig.geospatial_vertical_range_top_troposphere)
dataset.setncattr('geospatial_vertical_range_bottom_stratosphere', dataset_orig.geospatial_vertical_range_bottom_stratosphere)
dataset.setncattr('processor_version', dataset_orig.processor_version)
dataset.setncattr('product_version', dataset_orig.product_version)
dataset.setncattr('algorithm_version', dataset_orig.algorithm_version)

dataset.close()
dataset_orig.close()











# ''' Search for nc files '''
# for root, dirs, files in os.walk(search_folder):
#     # Loop through the file names
#     for file in files:
#         # Check if the file name contains "pepper"
#         if file.endswith('.nc'):
#             dict_ = Extract_netCDF4(os.path.join(root, file),
#                                     ['latitude_ccd', 'longitude_ccd', 'time', 'ozone_tropospheric_vertical_column', 'dates_for_tropospheric_column', 'qa_value'],
#                                     groups='all',
#                                     print_sum=False)
            
#             #print(dict_['dates_for_tropospheric_column'])

#             #dates = dict_['dates_for_tropospheric_column'].split(' ')
            
#             dataset_orig = nc.Dataset(os.path.join(root, file))
#             dates = [dataset_orig.time_coverage_start[:10],
#                      dataset_orig.time_coverage_end[:10]]
#             print(dates)
#             #--------------------------------------------------------
#             valid_lat_idx = []
#             for lat_idx in range(np.shape(dict_['latitude_ccd'])[0]):
#                 if (min_lat <= dict_['latitude_ccd'][lat_idx] and dict_['latitude_ccd'][lat_idx] <= max_lat):
#                     valid_lat_idx.append(int(lat_idx))

#             valid_lon_idx = []
#             for lon_idx in range(np.shape(dict_['longitude_ccd'])[0]):
#                 if (min_lon <= dict_['longitude_ccd'][lon_idx] and dict_['longitude_ccd'][lon_idx] <= max_lon):
#                     valid_lon_idx.append(int(lon_idx))
#             #--------------------------------------------------------
#             filtered_O3 = np.empty((len(valid_lat_idx), len(valid_lon_idx)), float)
            
#             for i in range(len(valid_lat_idx)):
#                 for j in range(len(valid_lon_idx)):
#                     filtered_O3[i, j] = dict_['ozone_tropospheric_vertical_column'][0, valid_lat_idx[i], valid_lon_idx[j]]
#             #--------------------------------------------------------
#             filtered_qa = np.empty((len(valid_lat_idx), len(valid_lon_idx)), float)
            
#             for i in range(len(valid_lat_idx)):
#                 for j in range(len(valid_lon_idx)):
#                     filtered_qa[i, j] = dict_['qa_value'][0, valid_lat_idx[i], valid_lon_idx[j]]
#             #--------------------------------------------------------
#             filename = file[:19]+'_'+dates[0]+'_'+dates[-1]+'_filt.nc'
#             #print(file[:19])

#             dataset = nc.Dataset(os.path.join(save_folder, filename), 'w')

#             lat_dim = dataset.createDimension('lat', len(valid_lat_idx))
#             lon_dim = dataset.createDimension('lon', len(valid_lon_idx))
#             time_dim = dataset.createDimension('time', len(dict_['dates_for_tropospheric_column']))

#             lat_var = dataset.createVariable('lat', 'f4', ('lat',))
#             lon_var = dataset.createVariable('lon', 'f4', ('lon',))
#             time_var = dataset.createVariable('dates_for_tropospheric_column', 'S'+str(len(dict_['dates_for_tropospheric_column'])), ('time',))
#             data_var = dataset.createVariable('ozone_tropospheric_vertical_column', 'f4', ('lat', 'lon',))
#             qa_var   = dataset.createVariable('qa_value', 'f4', ('lat', 'lon',))

#             lat_var.units  = 'degrees_north'
#             lon_var.units  = 'degrees_east'
#             time_var.units = 'YYYYMMDD'
#             data_var.units = 'mol m-2'
#             qa_var.units   = '%'
#             #lat_var[:]     = [dict_['latitude_ccd'][idx] for idx in valid_lat_idx]
#             lat_var[:]     = np.flip([dict_['latitude_ccd'][idx] for idx in valid_lat_idx])

#             lon_var[:]     = [dict_['longitude_ccd'][idx] for idx in valid_lon_idx]
#             time_var[:]    = nc.stringtochar(np.array([dict_['dates_for_tropospheric_column']], dtype='S'+str(len(dict_['dates_for_tropospheric_column']))))
#             #data_var[:, :] = filtered_O3
#             data_var[:, :] = np.flip(filtered_O3, axis=0)
#             qa_var[:, :]   = np.flip(filtered_qa, axis=0)

#             dataset.Conventions = 'CF-1.8'
#             dataset.title = 'Tropospheric ozone column. Start date: '+dates[0]+' end date: '+dates[-1]
#             dataset.institution = 'Lancaster University'
#             dataset.source = "Josh Miller - created by: spat_filter_L2__O3_TCL.py\nsource file: "+file+"\nData originally from Sentinel-5 precursor/TROPOMI Level 2 Product O3 Tropospheric Column (L2__O3_TCL)"
#             dataset.description = "Raw data from L2__O3_TCL has been spatially filtered to lat: ["+str(min_lat)+", "+str(max_lat)+"], lon: ["+str(min_lon)+", "+str(max_lon)+"]."
#             dataset.history = 'Created on '+str(date.today())
#             dataset.setncattr('crs', 'EPSG:4326')
#             dataset.setncattr('time_reference', dataset_orig.time_reference)
#             dataset.setncattr('time_coverage_start', dataset_orig.time_coverage_start)
#             dataset.setncattr('time_coverage_end', dataset_orig.time_coverage_end)
#             dataset.setncattr('time_coverage_troposphere_start', dataset_orig.time_coverage_troposphere_start)
#             dataset.setncattr('time_coverage_troposphere_end', dataset_orig.time_coverage_troposphere_end)
#             dataset.setncattr('geospatial_vertical_range_top_troposphere', dataset_orig.geospatial_vertical_range_top_troposphere)
#             dataset.setncattr('geospatial_vertical_range_bottom_stratosphere', dataset_orig.geospatial_vertical_range_bottom_stratosphere)
#             dataset.setncattr('processor_version', dataset_orig.processor_version)
#             dataset.setncattr('product_version', dataset_orig.product_version)
#             dataset.setncattr('algorithm_version', dataset_orig.algorithm_version)

#             dataset.close()

#             print("Saved:", 'S5P_RPRO_L2__O3_TCL_'+dates[0]+'_'+dates[-1]+'_filt.nc')