import iris
import netCDF4 as nc
import os
import sys
import numpy as np
iris.FUTURE.save_split_attrs = True

sys.path.append(os.getcwd())
from misc.misc_utils import GetDateInStr
from data_utils.extraction_funcs import Extract_netCDF4
#====================================================================
''' Merge files into one .nc file '''
# files = ['/Users/joshuamiller/Documents/Lancaster/Data/Vwind-ERA5/TEST_Vwind_l=135_2018-04-28_2018-12-31_14:00:00.grib',
#          '/Users/joshuamiller/Documents/Lancaster/Data/Vwind-ERA5/TEST_Vwind_l=135_2019-01-01_2019-06-30_14:00:00.grib',
#          '/Users/joshuamiller/Documents/Lancaster/Data/Vwind-ERA5/TEST_Vwind_l=135_2019-07-01_2019-12-31_14:00:00.grib',
#          '/Users/joshuamiller/Documents/Lancaster/Data/Vwind-ERA5/TEST_Vwind_l=135_2020-01-01_2020-06-30_14:00:00.grib', 
#          '/Users/joshuamiller/Documents/Lancaster/Data/Vwind-ERA5/TEST_Vwind_l=135_2020-07-01_2020-12-31_14:00:00.grib',
#          '/Users/joshuamiller/Documents/Lancaster/Data/Vwind-ERA5/TEST_Vwind_l=135_2021-01-01_2021-06-30_14:00:00.grib',
#          '/Users/joshuamiller/Documents/Lancaster/Data/Vwind-ERA5/TEST_Vwind_l=135_2021-07-01_2021-12-31_14:00:00.grib',
#          '/Users/joshuamiller/Documents/Lancaster/Data/Vwind-ERA5/TEST_Vwind_l=135_2022-01-01_2022-06-30_14:00:00.grib',
#          '/Users/joshuamiller/Documents/Lancaster/Data/Vwind-ERA5/TEST_Vwind_l=135_2022-07-01_2022-12-31_14:00:00.grib']

# files = ['/Users/joshuamiller/Documents/Lancaster/Data/Uwind-ERA5/ERA5_p=Uwind_l=135_2018-04-28_2022-12-31_14:00:00.grib']

files = []
for file in os.listdir("/Users/joshuamiller/Documents/Lancaster/Data/Temp-ERA5"):
    if file.endswith('.grib'):
        files.append(os.path.join("/Users/joshuamiller/Documents/Lancaster/Data/Temp-ERA5", file))
files.sort()

dest_file = 'AHHHHHH' # Put in scope

if (len(files) > 1):
    cubes_list = iris.load(files)
    combined_cube = iris.cube.CubeList(cubes_list).concatenate_cube()

    dest_file = '/Users/joshuamiller/Documents/Lancaster/Data/Temp-ERA5/ERA5_p=temp_l=137_2018-04-28_2022-12-31_14:00:00.nc'
    iris.save(combined_cube, dest_file)

else:
    print(len(files))
    cube = iris.load(files)

    dest_file = files[0][:-5]+'.nc'
    iris.save(cube, dest_file)
#====================================================================
''' Open the newly created .nc and do housekeeping '''
nc_file = nc.Dataset(dest_file, 'a')

print(nc_file.variables)
nc_file.source = 'Created by Joshua Miller - Lancaster University\nData originally from European Centre for Medium Range Weather Forecasts ERA5'

vars_ = ['latitude', 'longitude', 'air_temperature', 'forecast_reference_time', 'time']
units = ['degrees_north', 'degrees_east', 'degrees_Kelvin', 'hours since 1970/01/01 00:00:00', 'hours since 1970/01/01 00:00:00']
for i in range(len(vars_)):
    temp_var = nc_file.variables[vars_[i]]
    temp_var.units = units[i]

lon = nc_file.variables['longitude']
lon = lon[:]
if (180 <= max(lon)):
    lon -= 360 
nc_file.variables['longitude'] = lon

nc_file.close()

# dict_ = Extract_netCDF4("/Users/joshuamiller/Documents/Lancaster/Data/Whole_Area/Vwind/ERA5_p=Vwind_l=135_2018-04-28_2022-12-31_14:00:00.nc",
#                         var_names=['latitude', 'longitude', 'y_wind'],
#                         groups='all',
#                         print_sum=True)

# temp = np.squeeze(dict_['y_wind'])
# print("AHHHHHHHHHHHHH", np.shape(temp))