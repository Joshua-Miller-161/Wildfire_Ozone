import matplotlib.pyplot as plt
import geopandas as gpd
import netCDF4 as nc
import numpy as np
import sys
import os
from datetime import date

sys.path.append(os.getcwd())
from data_utils.preprocessing_funcs import Scale
from data_utils.extraction_funcs import Extract_netCDF4
#====================================================================
''' Folders '''
search_folder = "/Users/joshuamiller/Documents/Lancaster/Data/Ozone-L2_O3_TCL"
save_folder   = "/Users/joshuamiller/Documents/Lancaster/Data/Filtered_L2_O3_TCL"
#====================================================================
min_lat = -20
max_lat = 20
min_lon = -20
max_lon = 60
#====================================================================
''' Search for nc files '''
for root, dirs, files in os.walk(search_folder):
    # Loop through the file names
    for file in files[0:2]:
        # Check if the file name contains "pepper"
        if file.endswith('.nc'):
            dict_ = Extract_netCDF4(os.path.join(root, file),
                                    ['latitude_ccd', 'longitude_ccd', 'time', 'ozone_tropospheric_vertical_column', 'dates_for_tropospheric_column'],
                                    groups='all',
                                    print_sum=False)
            
            #print(dict_['dates_for_tropospheric_column'])

            dates = dict_['dates_for_tropospheric_column'].split(' ')
            
            #--------------------------------------------------------
            valid_lat_idx = []
            for lat_idx in range(np.shape(dict_['latitude_ccd'])[0]):
                if (min_lat <= dict_['latitude_ccd'][lat_idx] and dict_['latitude_ccd'][lat_idx] <= max_lat):
                    valid_lat_idx.append(int(lat_idx))

            valid_lon_idx = []
            for lon_idx in range(np.shape(dict_['longitude_ccd'])[0]):
                if (min_lon <= dict_['longitude_ccd'][lon_idx] and dict_['longitude_ccd'][lon_idx] <= max_lon):
                    valid_lon_idx.append(int(lon_idx))
            #--------------------------------------------------------
            filtered_O3 = np.empty((len(valid_lat_idx), len(valid_lon_idx)), float)
            
            for i in range(len(valid_lat_idx)):
                for j in range(len(valid_lon_idx)):
                    filtered_O3[i, j] = dict_['ozone_tropospheric_vertical_column'][0, valid_lat_idx[i], valid_lon_idx[j]]
            #--------------------------------------------------------
            filename = file[:19]+'_'+dates[0]+'-'+dates[-1]+'_filt.nc'
            #print(file[:19])

            dataset = nc.Dataset(os.path.join(save_folder, filename), 'w')

            lat_dim = dataset.createDimension('lat', len(valid_lat_idx))
            lon_dim = dataset.createDimension('lon', len(valid_lon_idx))
            time_dim = dataset.createDimension('time', len(dict_['dates_for_tropospheric_column']))

            lat_var = dataset.createVariable('lat', 'f4', ('lat',))
            lon_var = dataset.createVariable('lon', 'f4', ('lon',))
            time_var = dataset.createVariable('dates_for_tropospheric_column', 'S'+str(len(dict_['dates_for_tropospheric_column'])), ('time',))
            data_var = dataset.createVariable('ozone_tropospheric_vertical_column', 'f4', ('lat', 'lon',))

            lat_var.units = 'degrees_north'
            lon_var.units = 'degrees_east'
            time_var.units = 'YYYYMMDD'
            data_var.units = 'Dobson_units'

            #lat_var[:]     = [dict_['latitude_ccd'][idx] for idx in valid_lat_idx]
            lat_var[:]     = np.flip([dict_['latitude_ccd'][idx] for idx in valid_lat_idx])

            lon_var[:]     = [dict_['longitude_ccd'][idx] for idx in valid_lon_idx]
            time_var[:]    = nc.stringtochar(np.array([dict_['dates_for_tropospheric_column']], dtype='S'+str(len(dict_['dates_for_tropospheric_column']))))
            #data_var[:, :] = filtered_O3
            data_var[:, :] = np.flip(filtered_O3, axis=0)

            dataset.Conventions = 'CF-1.8'
            dataset.title = 'Tropospheric ozone column. Start date: '+dates[0]+' end date: '+dates[-1]
            dataset.institution = 'Lancaster University'
            dataset.source = 'Josh Miller - created by: spat_filter_L2__O3_TCL.py - source file:\n'+file
            dataset.history = 'Created on '+str(date.today())

            dataset.close()

            print("Saved:", 'S5P_RPRO_L2__O3_TCL_'+dates[0]+'-'+dates[-1]+'_filt.nc')