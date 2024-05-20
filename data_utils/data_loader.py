import sys
sys.dont_write_bytecode = True
import numpy as np
import os
import yaml
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(os.getcwd())
from data_utils.extraction_funcs import Extract_netCDF4
from data_utils.preprocessing_funcs import Scale
#====================================================================
def DataLoader(config_path,
               data_config_path,
               data_type):
    #----------------------------------------------------------------
    ''' Get variables from config '''

    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    region = config['REGION']
    assert (region in ['Whole_Area', 'East_Ocean', 'West_Ocean', 'South_Land', 'North_Land']), region+" must be: 'Whole_Area', 'East_Ocean', 'West_Ocean', 'South_Land', 'North_Land'"
    
    assert (data_type in ['HISTORY_DATA', 'TARGET_DATA'])

    variables = config[data_type]
    for var in variables:
        assert (var in ['ozone', 'fire', 'temp', 'u-wind', 'v-wind', 'lon', 'lat', 'time']), var+" must be: 'ozone', 'fire', 'temp', 'u-wind', 'v-wind', 'lon', 'lat', 'time'"

    start_date = datetime.strptime(config['TIME_WINDOW']['start_date'], '%Y-%m-%d')
    end_date   = datetime.strptime(config['TIME_WINDOW']['end_date'], '%Y-%m-%d')
    assert (datetime(2018, 4, 30) <= start_date and start_date <= datetime(2022, 7, 31)), "'start_date' must be between 2018-04-30 and 2022-07-31"
    assert (datetime(2018, 4, 30) <= end_date and end_date <= datetime(2022, 7, 31)), "'end_date' must be between 2018-04-30 and 2022-07-31"
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    with open(data_config_path, 'r') as c:
        data_config = yaml.load(c, Loader=yaml.FullLoader)

    data_path = data_config['DATA_FOLDER']
    
    scale_types = data_config['PREPROCESSING'][data_type]
    print("ASDASDASDASDADS", type(scale_types), list(scale_types.values()))

    for scale_type in list(scale_types.values()):
        print(scale_type)
        if (type(scale_type) == str):
            assert (scale_type in ['none', 'minmax', 'standard', 'log']), var+" must be: 'none', 'minmax', 'standard', 'log'"

    ozone_scale_type = data_config['PREPROCESSING'][data_type]['ozone_scale_type']
    fire_scale_type  = data_config['PREPROCESSING'][data_type]['fire_scale_type']
    temp_scale_type  = data_config['PREPROCESSING'][data_type]['temp_scale_type']
    uwind_scale_type = data_config['PREPROCESSING'][data_type]['uwind_scale_type']
    vwind_scale_type = data_config['PREPROCESSING'][data_type]['vwind_scale_type']
    lat_scale_type   = data_config['PREPROCESSING'][data_type]['lat_scale_type']
    lon_scale_type   = data_config['PREPROCESSING'][data_type]['lon_scale_type']
    time_scale_type  = data_config['PREPROCESSING'][data_type]['time_scale_type']
    scale_factor     = data_config['PREPROCESSING'][data_type]['scale_factor']
    #----------------------------------------------------------------
    ''' Get folders to the relevant data '''

    base_path    = os.path.join(data_path, region)
    folder_names = {'ozone': 'Ozone',
                    'fire': 'Fire',
                    'temp': 'Temp',
                    'u-wind': 'Uwind',
                    'v-wind': 'Vwind'}
    
    data_paths_dict = {}
    for var in variables:
        if var in folder_names.keys():
            var_path = os.path.join(base_path, folder_names[var])

            files = os.listdir(var_path)
            for file in files:
                if (file.endswith('.nc') and ('2018-04-30' in file) and ('2022-07-31' in file)):
                    data_paths_dict[var] = os.path.join(var_path, file)

    #print(data_paths_dict)
    print("variables", variables)
    #----------------------------------------------------------------
    ''' Prepare to extract data '''

    var_names_dict = {'ozone' : ['lat', 'lon', 'start_date', 'ozone_tropospheric_vertical_column'],
                      'fire'  : ['lat', 'lon', 'date', 'frp'],
                      'temp'  : ['lat', 'lon', 'time', 'air_temperature'],
                      'u-wind': ['lat', 'lon', 'time', 'x_wind'],
                      'v-wind': ['lat', 'lon', 'time', 'y_wind'],
                      'lat'   : 'lat',
                      'lon'   : 'lon',
                      'time'  : 'start_date'}
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Initialize final array

    dict_ = Extract_netCDF4(data_paths_dict['ozone'],
                            var_names=[var_names_dict['ozone'][0], 
                                       var_names_dict['ozone'][1]],
                            groups='all',
                            print_sum=False)
    
    num_days = (end_date-start_date).days+1
    lat_dim  = np.shape(dict_[var_names_dict['ozone'][0]])[0]
    lon_dim  = np.shape(dict_[var_names_dict['ozone'][1]])[0]

    data = np.ones((num_days, lat_dim, lon_dim, len(variables)), float) * -999
    print("data", np.shape(data))
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Filter time

    data_start_date = datetime(2018, 4, 30)

    start_offset = (start_date - data_start_date).days
    end          = (end_date - data_start_date).days
    valid_date_idx = np.arange(start_offset, end+1)
    print("start_offset =", start_offset, end, np.shape(valid_date_idx))
    #----------------------------------------------------------------
    ''' Loop through desired variables and extract data '''

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    for idx in range(len(variables)):
        var = variables[idx]
        print("Processing:", var,'\n')

        dict_ = -999
        data_ = -999
        if (var in ['ozone', 'fire', 'temp', 'u-wind', 'v-wind']):
            print("Source:", data_paths_dict[var])

            dict_ = Extract_netCDF4(data_paths_dict[var],
                                    var_names=[var_names_dict[var][3]],
                                    groups='all')
            data_ = np.squeeze(np.asarray(dict_[var_names_dict[var][3]]))

        elif (var in ['lat', 'lon', 'time']):
            print("Source:", data_paths_dict['ozone'])

            dict_ = Extract_netCDF4(data_paths_dict['ozone'],
                                    var_names=[var_names_dict[var]],
                                    groups='all')
            data_ = np.squeeze(np.asarray(dict_[var_names_dict[var]]))
            
            # time, lat, lon
            if (var == 'lat'):
                #print("lat:", np.shape(data_), data_)
                data_ = np.tile(np.reshape(data_, (-1, 1)), ((datetime(2022, 7, 31)-datetime(2018, 4, 30)).days+1, 1, lon_dim))
                #print("new lat:", np.shape(data_), '\n', data_)
            
            elif (var == 'lon'):
                #print("lon:", np.shape(data_), data_)
                data_ = np.tile(data_, ((datetime(2022, 7, 31)-datetime(2018, 4, 30)).days+1, lat_dim, 1))
                #print("new lon:", np.shape(data_), '\n', data_)

            elif (var == 'time'):
                #print("time:", np.shape(data_), data_)
                data_ = np.tile(np.reshape(data_, (-1, 1, 1)), (1, lat_dim, lon_dim))
                #print("new time:", np.shape(data_), '\n', data_)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Temporally filter data

        filtered_data = np.ones((num_days, np.shape(data_)[1], np.shape(data_)[2]), float) * -999
        num_skipped = 0
        for i in range(np.shape(data_)[0]):
            if ((start_offset <= i) and (i <= end)):
                filtered_data[i-num_skipped] = data_[i]
            else:
                num_skipped += 1
        #print(np.shape(data_), np.shape(filtered_data))
        print("Shape:", np.shape(filtered_data))
        del(data_)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Scale data

        if (var == 'ozone'):
            if not (ozone_scale_type == 'none'):
                filtered_data = Scale(filtered_data, ozone_scale_type, data_name='ozone')
        elif (var == 'fire'):
            if not (fire_scale_type == 'none'):
                filtered_data = Scale(filtered_data, fire_scale_type, scale_factor, data_name='fire')
        elif (var == 'temp'):
            if not (temp_scale_type == 'none'):
                filtered_data = Scale(filtered_data, temp_scale_type, data_name='temp')
        elif (var == 'u-wind'):
            if not (uwind_scale_type == 'none'):
                filtered_data = Scale(filtered_data, uwind_scale_type, data_name='u-wind')
        elif (var == 'v-wind'):
            if not (vwind_scale_type == 'none'):
                filtered_data = Scale(filtered_data, vwind_scale_type, data_name='v-wind')
        elif (var == 'lat'):
            if not (lat_scale_type == 'none'):
                filtered_data = Scale(filtered_data, lat_scale_type, data_name='lat')
        elif (var == 'lon'):
            if not (lon_scale_type == 'none'):
                filtered_data = Scale(filtered_data, lon_scale_type, data_name='lon')
        elif (var == 'time'):
            if not (time_scale_type == 'none'):
                filtered_data = Scale(filtered_data, time_scale_type, data_name='time')
        print("----------------------------------------------------")
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        data[:, :, :, idx] = filtered_data
    #----------------------------------------------------------------
    return data
#====================================================================
# fig, ax = plt.subplots(1, 2, figsize=(7,7))
# data = DataLoader('config.yml', 'HISTORY_DATA')

# with open('config.yml', 'r') as c:
#     config = yaml.load(c, Loader=yaml.FullLoader)
# variables = config['HISTORY_DATA']

# var_idx = 1
# time_idx = 100

# data_transposed = np.transpose(data, (3, 0, 1, 2))
# data_reshaped = data_transposed.reshape(data_transposed.shape[0], -1)

# print("data:", np.shape(data), ", trans", np.shape(data_transposed), ", reshape", np.shape(data_reshaped))

# lol1 = ax[0].imshow(data[time_idx, :, :, var_idx])

# divider = make_axes_locatable(ax[0])
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cbar = plt.colorbar(lol1, cax=cax, label='')
# cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=180)
# ax[0].set_title(variables[var_idx])


# lol1 = ax[1].imshow(data_transposed[var_idx, time_idx, :, :])

# divider = make_axes_locatable(ax[1])
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cbar = plt.colorbar(lol1, cax=cax, label='')
# cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=180)
# ax[1].set_title(variables[var_idx])

# plt.show()