import iris
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
import sys
import shutil
import netCDF4 as nc

sys.path.append(os.getcwd())
from data_utils.extraction_funcs import ExtractGRIBIris, Extract_netCDF4
from misc.misc_utils import GetDateInStr
#====================================================================
def find_elements_not_in_list(reference, list1):
    # Convert both lists to sets for efficient membership checking
    reference_set = set(reference)
    list1_set = set(list1)
    
    # Find elements in reference that are not in list1
    result = reference_set - list1_set
    
    return list(result)
#====================================================================
start_date = datetime(2018, 4, 30)
end_date   = datetime(2022, 7, 31)

complete_dates = [start_date + timedelta(days=i) for i in range((end_date-start_date).days+1)]
#====================================================================
file = '/Users/joshuamiller/Documents/Lancaster/Data/Whole_Area/Ozone/S5P_RPRO_L2_O3__TCL_2018-04-30_2022-07-31_Whole_Area.nc'
#file = '/Users/joshuamiller/Documents/Lancaster/Data/Uwind-ERA5/ERA5_p=Uwind_l=135_2018-04-28_2022-12-31_14:00:00.grib'

folder = "/Users/joshuamiller/Documents/Lancaster/Data/L2_O3_TCL"

dates_list = []
for root, dirs, files in os.walk(folder):
    # Loop through the file names
    for file in files:
        # Check if the file name contains "pepper"
        if file.endswith('.nc'):
            dataset_orig = nc.Dataset(os.path.join(root, file))
            
            dates_list.append(datetime.strptime(dataset_orig.time_coverage_start[:10], '%Y-%m-%d'))
            
            dataset_orig.close()

dates_list.sort()
print(len(dates_list), len(complete_dates))
missing = find_elements_not_in_list(complete_dates, dates_list)
missing.sort()

for date in missing:
     print(date)

            # dict_ = Extract_netCDF4(os.path.join(root, file),
            #                         ['latitude_ccd', 'longitude_ccd', 'sensing_time', 'time', 'ozone_tropospheric_vertical_column', 'dates_for_tropospheric_column'],
            #                         groups='all',
            #                         print_sum=True)
            
            # print(dict_['sensing_time'])


            # dates = dict_['dates_for_tropospheric_column'].split(' ')





# if file.endswith('.nc'):
#     dict_ = Extract_netCDF4(file, 
#                             var_names=['start_date'],
#                             groups='all',
#                             print_sum=True)
#     # hours = dict_['forecast_reference_time']
#     # dates_nc = [datetime(1970, 1, 1, 0, 0, 0) + timedelta(hours=hour) for hour in hours]
    
#     days = dict_['start_date']
#     dates_nc = [datetime(1970, 1, 1, 0, 0, 0) + timedelta(days=day) for day in days]

#     print("len(dates_nc)=", len(dates_nc), ", len(complete_dates)=", len(complete_dates))

#     eq = complete_dates == dates_nc
#     if not eq:
#         missing = find_elements_not_in_list(complete_dates, dates_nc)
#         missing.sort()

# for date in missing:
#     print(date)



# elif file.endswith('.grib'):
#     dict_ = ExtractGRIBIris(file,
#                             var_names='all',
#                             sclice_over_var='model_level_number',
#                             print_sum=True,
#                             num_examples=1)
    
#     dates_str = dict_['forecast_reference_time']
    
#     dates_grib = [datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') for date_str in dates_str]
#     print(dates_grib)

#     eq = complete_dates == dates_grib

#     print("len(dates_grib)=", len(dates_grib), ", len(complete_dates)=", len(complete_dates), eq)

#     if not eq:
#         print(find_elements_not_in_list(complete_dates, dates_grib))

#====================================================================
# for folder_ in os.listdir(folder):
#     if os.path.isdir(os.path.join(folder, folder_)) and folder_.endswith(" 3"):
#         # Construct full folder path
#         folder_path = os.path.join(folder, folder_)
#         try:
#             # Delete the folder
#             shutil.rmtree(folder_path)
#             print(f"Deleted folder: {folder_path}")
#         except Exception as e:
#             print(f"Failed to delete folder: {folder_path}. Reason: {e}")



#====================================================================