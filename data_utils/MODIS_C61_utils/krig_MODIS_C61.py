import numpy as np
from scipy.spatial import KDTree
import os
import sys
import pandas as pd
from datetime import datetime

sys.path.append(os.getcwd())
from data_utils.extraction_funcs import Extract_netCDF4
#====================================================================
''' Folders '''
search_folder = '/Users/joshuamiller/Documents/Lancaster/Data/Fire-MODIS_C61'
save_folder   = '/Users/joshuamiller/Documents/Lancaster/Data/Kriged_MODIS_C61'
#====================================================================
''' Get reference lat/lon grid & make tree '''
dict_ = Extract_netCDF4('/Users/joshuamiller/Documents/Lancaster/Data/Kriged_L2_O3_TCL/S5P_OFFL_L2__O3_TCL_20180526-20180529_kriged_199.nc',
                        ['lat', 'lon'],
                        groups='all',
                        print_sum=False)
O3_lat = dict_['lat']
O3_lon = dict_['lon']

O3_lat_tiled = np.tile(O3_lat, (np.shape(O3_lon)[0], 1)).T
O3_lon_tiled = np.tile(O3_lon, (np.shape(O3_lat)[0], 1))

O3_latlon_points = np.array(list(zip(O3_lat_tiled.ravel(), O3_lon_tiled.ravel())))
tree = KDTree(O3_latlon_points)
#====================================================================
''' Search through files and krig where necessary '''
frp_conf_dict = {'lat': O3_latlon_points[:, 0],
                 'lon': O3_latlon_points[:, 1],
                 'frp': np.zeros_like(O3_latlon_points[:, 0]),
                 'confidence': np.zeros_like(O3_latlon_points[:, 0])}

for file in os.listdir(search_folder):
    if file.endswith('.csv'):
        frp_conf_df = pd.DataFrame.from_dict(frp_conf_dict)
        print("- + - + - + - + - + - + - + - + - + - + - + - + - + - + -")
        print(">> opening:", file)
        #------------------------------------------------------------
        ''' Get data '''
        fire_df = pd.read_csv(os.path.join(search_folder, file))

        #------------------------------------------------------------
        ''' Perform kriging '''
        for i in range(fire_df.shape[0]):
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            ''' Get num_neighbors valid points nearest invalid point '''
            point = np.array([fire_df.loc[i, 'latitude'], fire_df.loc[i, 'longitude']])
            
            dis, idx = tree.query(point, k=1)
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            ''' Krig '''
            # print("i =", i, ", idx =", idx, ", frp =", fire_df.loc[i, 'frp'], ", point =", point, ", nearest =", O3_latlon_points[idx, :], ", dist =", round(dis, 3))

            if (frp_conf_df.loc[idx, 'confidence'] < fire_df.loc[i, 'confidence']):
                frp_conf_df.loc[idx, 'frp'] = fire_df.loc[i, 'frp']
                frp_conf_df.loc[idx, 'confidence'] = fire_df.loc[i, 'confidence']
            # else:
            #     print("ALEADY BIGGER i =", i, ", idx =", idx, ", prev. =", frp_conf_df.loc[idx, 'frp'], ", new =", fire_df.loc[i, 'frp'])
        #------------------------------------------------------------
        ''' Save the kriged file '''
        date_ = str(fire_df.loc[0, 'acq_date'])

        filename = 'MODIS_C61_'+date_+'_kriged.csv'
        
        frp_conf_df.to_csv(os.path.join(save_folder, filename), index=False)

        print(">> saved:", filename)