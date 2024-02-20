import numpy as np
import netCDF4 as nc
from scipy.spatial import KDTree
from data_utils.preprocessing_funcs import Scale, DoKrig
from data_utils.extraction_funcs import Extract_netCDF4
#====================================================================
''' Number of neighbors '''
num_neighbors = 100
#====================================================================
''' Folders '''
search_folder = '/Users/joshuamiller/Documents/Lancaster/Data/Filtered_L2_O3_TCL'
save_folder   = '/Users/joshuamiller/Documents/Lancaster/Data/Kriged_L2_O3_TCL_'+str(num_neighbors)
#====================================================================
''' Search through files and krig where necessary '''
for file in os.listdir(search_folder):
    if file.endswith('.nc'):
        print("- + - + - + - + - + - + - + - + - + - + - + - + - + - + -")
        print(">> opening:", file)
        #------------------------------------------------------------
        ''' Get data '''
        dict_ = Extract_netCDF4(os.path.join(search_folder, file),
                                ['lat', 'lon', 'ozone_tropospheric_vertical_column', 'dates_for_tropospheric_column'],
                                groups='all',
                                print_sum=False)

        dates_str  = ''.join(dict_['dates_for_tropospheric_column'])
        dates_list = dates_str.split(' ')

        lat = dict_['lat']
        lon = dict_['lon']
        O3  = dict_['ozone_tropospheric_vertical_column']
        O3_shape = np.shape(O3)
        mask = np.ma.getmaskarray(O3)
        mask_ = mask.ravel()
        
        # print("lon:", lon)
        # print("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
        # print("lat:", lat)
        # print("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
        lon_tiled = np.tile(lon, (np.shape(lat)[0], 1))
        # print("lon_tiled:", lon_tiled)
        # print("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
        lat_tiled = np.tile(lat, (np.shape(lon)[0], 1)).T
        # print("lat_tiled:", lat_tiled)
        # print("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
        latlon_points = np.array(list(zip(lat_tiled.ravel(), lon_tiled.ravel())))
        # print("ozone:", O3)
        # print("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
        O3_points = O3.ravel()

        #print('lon:', lon_tiled[1][0], ' lat:', lat_tiled[1][0], ' O3:', O3[1][0])
        #print('lon:', latlon_points[1][0], ' lat:', latlon_points[0][1], ' O3:', O3_points[1])
        #print("dates =", dates, np.shape(lat), type(lat), np.shape(lon), np.shape(lat_tiled), np.shape(lon_tiled), 
        #      np.shape(latlon_points), np.shape(O3_points))
        #print("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
        #print("   lat,   lon,   O3")
        #for i in range(4):
        #    print(latlon_points[i, :], O3_points[i])
        #------------------------------------------------------------
        ''' Create tree. Excludes locations with inval'''
        invalid_locs = []
        invalid_locs_idx = []
        valid_locs = []
        valid_locs_idx = []

        for i in range(np.shape(latlon_points)[0]):
            if (math.isnan(O3_points[i]) or np.isnan(O3_points[i]) or (mask_[i] == True)):
                invalid_locs.append(latlon_points[i, :])
                invalid_locs_idx.append(i)
            else:
                valid_locs.append(latlon_points[i, :])
                valid_locs_idx.append(i)

        invalid_locs = np.asarray(invalid_locs)
        invalid_locs_idx = np.asarray(invalid_locs_idx)
        valid_locs = np.asarray(valid_locs)
        valid_locs_idx = np.asarray(valid_locs_idx)
        print("==========================")
        print(np.shape(invalid_locs_idx)[0], 'invalid points out of', np.shape(latlon_points)[0], round(100* np.shape(invalid_locs_idx)[0] / np.shape(latlon_points)[0], 3), "percent")
        print("==========================")
        tree = KDTree(valid_locs)
        #------------------------------------------------------------
        ''' Perform kriging '''
        for i in range(np.shape(invalid_locs)[0]):
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            ''' Get num_neighbors valid points nearest invalid point '''
            point = invalid_locs[i, :]
            
            dis, idx = tree.query(point, k=num_neighbors)

            lon_krig = latlon_points[valid_locs_idx[idx], 1]
            lat_krig = latlon_points[valid_locs_idx[idx], 0]
            val_krig = O3_points[valid_locs_idx[idx]]
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            ''' Krig '''
            #print("i =", i, ", idx =", idx, ", val =", val_krig, ", point =", invalid_locs_idx[i], " lon =", lon_krig, ", lat =", lat_krig)
            zstar, ss = DoKrig(lon_krig, lat_krig, val_krig, point[0], point[1])

            O3_points[invalid_locs_idx[i]] = zstar[0]
            #print("--------------- est. ", zstar[0], ", acc ", O3_points[i],"------------------")
        #------------------------------------------------------------
        ''' Save the kriged file '''
        filename = file[:19]+'_'+dates_list[0]+'-'+dates_list[-1]+'_kriged_'+str(np.shape(invalid_locs)[0])+'.nc'
        #print(len(dates_str), file[:19])

        dataset = nc.Dataset(os.path.join(save_folder, filename), 'w')

        lat_dim = dataset.createDimension('lat', np.shape(lat)[0])
        lon_dim = dataset.createDimension('lon', np.shape(lon)[0])
        time_dim = dataset.createDimension('time', len(dates_str))

        lat_var = dataset.createVariable('lat', 'f4', ('lat',))
        lon_var = dataset.createVariable('lon', 'f4', ('lon',))
        time_var = dataset.createVariable('dates_for_tropospheric_column', 'S'+str(len(dates_str)), ('time',))
        data_var = dataset.createVariable('ozone_tropospheric_vertical_column', 'f4', ('lat', 'lon',))

        lat_var.units = 'degrees_north'
        lon_var.units = 'degrees_east'
        time_var.units = 'YYYYMMDD'
        data_var.units = 'Dobson_units'

        lat_var[:]     = lat
        lon_var[:]     = lon
        time_var[:]    = nc.stringtochar(np.array(dates_str, dtype='S'+str(len(dates_str))))
        data_var[:, :] = O3_points.reshape(O3_shape)

        dataset.Conventions = 'CF-1.8'
        dataset.title = 'Tropospheric ozone column. Start date: '+dates_list[0]+' end date: '+dates_list[-1]
        dataset.institution = 'Lancaster University'
        dataset.source = 'Josh Miller - created by: krig_L2__O3_TCL.py\nSource file: '+file+'\n'+'Kriged '+str(np.shape(invalid_locs)[0]) + ' of ' + str(np.shape(latlon_points)[0]) + ' points; '+str(num_neighbors) + ' used for kriging.'
        dataset.history = 'Created on '+str(date.today())

        dataset.close()

        print(">> saved:", filename)