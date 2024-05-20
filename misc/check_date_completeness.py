import numpy as np
import sys
import os
import datetime
import pandas as pd


sys.path.append(os.getcwd())
from data_utils.extraction_funcs import Extract_netCDF4
#====================================================================

def CheckDateCompleteness(base_path):
    rpro_dates = []
    offl_dates = []
    folders = os.listdir(base_path)

    for folder in folders:
        if not os.path.isdir(os.path.join(base_path, folder)):
            folders.remove(folder)
    #================================================================
    for folder in folders:
        files = os.listdir(os.path.join(base_path, folder))
        for file in files:
            if file.endswith('.nc'):
                date_dict = Extract_netCDF4(os.path.join(os.path.join(base_path, folder), file), 
                                                    var_names=['dates_for_tropospheric_column'], 
                                                    groups='all', 
                                                    print_sum=False)
                date = date_dict['dates_for_tropospheric_column'].split(' ')[0]
                if "RPRO" in file:
                    rpro_dates.append(date[0:4]+'-'+date[4:6]+'-'+date[6:8])
                elif "OFFL" in file:
                    offl_dates.append(date[0:4]+'-'+date[4:6]+'-'+date[6:8])
    #================================================================
    rpro_dates.sort()
    offl_dates.sort()

    date_list = pd.date_range(start=rpro_dates[0], end=rpro_dates[-1])
    date_list = date_list.to_pydatetime().tolist()

    # for i in range(len(rpro_dates)):
    #     rpro_dates[i] = rpro_dates[i].strftime('%Y-%m-%d')


    #print(rpro_dates)
    #print("--------------------------------------")
    #print(offl_dates)
    #================================================================
    missing_days = []
    for date in date_list:
        if not date in rpro_dates:
            missing_days.append(date)
    print("======================================")
    #print(missing_days)
    all_dates = rpro_dates + offl_dates
    all_dates.sort()
    del(rpro_dates)
    del(offl_dates)
    return all_dates
#====================================================================
x = CheckDateCompleteness("/Users/joshuamiller/Documents/Lancaster/Data/L2_O3_TCL")
print(x)


#====================================================================
# path1 = "/Users/joshuamiller/Documents/Lancaster/Data/L2_O3_TCL/S5P_OFFL_L2__O3_TCL_20200109T121423_20200115T130023_11607_01_010107_20200124T000055/S5P_OFFL_L2__O3_TCL_20200109T121423_20200115T130023_11607_01_010107_20200124T000055.nc"
# path2 = "/Users/joshuamiller/Documents/Lancaster/Data/L2_O3_TCL/S5P_OFFL_L2__O3_TCL_20200108T105152_20200114T131922_11592_01_010107_20200123T043622/S5P_OFFL_L2__O3_TCL_20200108T105152_20200114T131922_11592_01_010107_20200123T043622.nc"

# date_dict = Extract_netCDF4(path2, 
#                             var_names=['dates_for_tropospheric_column'], 
#                             groups='all', 
#                             print_sum=False)

# print(date_dict)
#====================================================================