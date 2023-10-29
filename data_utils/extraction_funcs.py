import netCDF4 as nc
import h5py
import numpy as np
import xarray as xr
import dask.array as da
import iris_grib
from osgeo import gdal
import rasterio
import csv
import sys
import os


sys.path.append(os.getcwd())
from misc.misc_utils import FindDate

def Extract_netCDF4(path, var_names, groups=None, print_sum=False):
    '''
    This will read the desired variables from a .nc file.
    
    Returns: A dict of dask arrays or numpy arrays of the variables desired
    
    Parameters:
    - path (str) - path to the nc file
    - var_names (list) - the names of the variable(s) to be extracted
    - group (list, optional) - list of the name(s) of the group in the nc file to be accessed.
                               Can also use 'all' to search all the groups
    - print_sum (Bool, optional) - print out a summary of the dataset
    '''
    assert path.endswith('.nc') or path.endswith('.nc4') or path.endswith('.nc4'), "File must be .nc"
    #----------------------------------------------------------------
    if not ((groups == None)  or (groups == 'all')):
        if not type(groups) == list:
            print('AHHHHHHHHHH', type(groups))
            groups = [groups]
    #----------------------------------------------------------------
    var_dict = {}
    #----------------------------------------------------------------
    ds = nc.Dataset(path, "r")
    if print_sum:
        print("SUMMARY :", ds)
    #----------------------------------------------------------------
    if groups == 'all':
        groups = list(ds.groups.keys())

    if print_sum:
        print("GROUPS:", groups)
        print("VARIABLES:", list(ds.variables.keys()))
        
        ds_gdal = gdal.Open(path)
        if not ds_gdal.GetProjectionRef() == '':
            print("CRS (gdal):", ds_gdal.GetProjectionRef())
            del(ds_gdal)
        else:
            ds_rasterio = rasterio.open(path)
            if not (ds_rasterio.crs == ''):
                print("CRS (rasterio):", ds_rasterio.crs)
            else:
                print("Couldn't get crs")

        

    #----------------------------------------------------------------
    valid_keys_list, valid_keys = GetKeysNC(ds)

    assert all(var_name in valid_keys_list for var_name in var_names), "Some variable names are not in the file. Valid variable names in this are:\n"+ valid_keys
    #----------------------------------------------------------------
    if groups==None:
        for var_name in var_names:
            var = ds.variables[var_name] # access the variable as a dataset object
            var_dict[var_name] = var[:]  # convert to numpy array using [:]
    #----------------------------------------------------------------
    else:
        for group in groups:
            group_ds = ds.groups[group]

            if print_sum:
                print(' - + - + - + - Examining group \''+group+'\' - + - + - + -')
                PrintSumNC(group_ds)
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if (len(list(group_ds.variables.keys())) > 0):
                for var_name in var_names:
                    if var_name in list(group_ds.variables.keys()):
                        var = group_ds.variables[var_name]
                        var_dict[var_name] = var[:]

            else:
                dict_of_group = ds.groups[group].__dict__
                for var_name in var_names:
                    if var_name in list(dict_of_group.keys()):
                        var = dict_of_group[var_name]
                        var_dict[var_name] = var
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    ds.close() # close the file
    return var_dict

def GetKeysNC(ds):
    '''
     - ds (nc.Dataset)
    '''
    valid_keys_list = []
    valid_keys_str = ''

    if (len(list(ds.groups.keys())) > 0): # Iterate through the froups
        for group in ds.groups.keys():
            if isinstance(ds.groups[group], nc._netCDF4.Group): # This 'key' is a group name
                if (len(list(ds.groups[group].variables.keys())) > 0):
                    for var in list(ds.groups[group].variables.keys()):  # Now iterating through variable names
                        valid_keys_list.append(var)
                        valid_keys_str += var + ', '
                else:
                    for var in list(ds.groups[group].__dict__.keys()):  # Now iterating through keys in the dictionary structure of the group
                        valid_keys_list.append(var)
                        valid_keys_str += var + ', '
            else:
                valid_keys_list.append(group)
                valid_keys_str += group + ', '

    else: # No groups
        for var in list(ds.variables.keys()):
            valid_keys_list.append(var)
            valid_keys_str += var + ', '
    
    return valid_keys_list, valid_keys_str

def PrintSumNC(ds):
    for name, var in ds.variables.items(): # loop through the variables
        print(name, var.shape)
    

#====================================================================
#====================================================================
#====================================================================
#====================================================================

def ExtractHDF5(path, var_names, groups=None, print_sum=False):
    '''
    This will read the desired variables from an hdf5 file.
    
    Returns: A dict of dask arrays or numpy arrays of the variables desired
    
    Parameters:
    - path (str) - path to the hdf5 file
    - var_names (list) - the names of the variable to be extracted
    - group (list, optional) - list of the name(s) of the group in the hdf5 file to be accessed
                               Can also use 'all' to search all the groups
    - print_sum (Bool, optional) - print out a summary of the dataset
    '''
    assert path.endswith('.hdf5') or path.endswidth('.h5'), "File must be .hdf5 or .h5. Got:\n"+path
    #----------------------------------------------------------------
    if not ((groups == None) or (groups == 'all')):
        if not type(groups) == list:
            print('AHHHHHHHHHH', type(groups))
            groups = [groups]
    #----------------------------------------------------------------
    if not (type(var_names) == list):
        var_names = [var_names]
    #----------------------------------------------------------------
    var_dict = {}
    #----------------------------------------------------------------
    f = h5py.File(path, "r") # open the file in read mode
    #----------------------------------------------------------------
    valid_keys_list, valid_keys_str = GetKeysHDF5(f)
    
    assert all(var_name in valid_keys_list for var_name in var_names), "Some variable names are not in the file. Valid variable names are:\n"+ valid_keys_str
    #----------------------------------------------------------------
    if (groups == 'all'):
        groups = list(f.keys())
    #----------------------------------------------------------------
    if (groups==None):
        if print_sum:
            PrintSumHDF5(f)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        for var_name in var_names:
            var = f[var_name] # access the variable as a dataset object
            var_dict[var_name] = var[()] # convert to numpy array using [()]
    #----------------------------------------------------------------
    else:
        for group in groups:
            if print_sum:
                print(' - + - + - + - Examining group \''+group+'\' - + - + - + -')
                PrintSumHDF5(f[group])
            for var_name in var_names:
                if var_name in list(f[group].keys()):
                    var = f[group+'/'+var_name]
                    var_dict[var_name] = var[()]
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    f.close() # close the file
    return var_dict

def GetKeysHDF5(f):
    '''
    
     - f (h5py.Dataset)
    '''
    valid_keys_list = []
    valid_keys_str = ''

    for key in f.keys():
        #print("+++++++++++", type(f[key]), "++++++++++++")
        if isinstance(f[key], h5py._hl.group.Group): # This 'key' is a group name
            for var in f[key].keys():  # Now iterating through variable names
                valid_keys_list.append(var)
                valid_keys_str += var + ', '
        else:
            valid_keys_list.append(key)
            valid_keys_str += key + ', '

    return valid_keys_list, valid_keys_str

def PrintSumHDF5(f):
    '''
    Prints the variable name, shape
     - f (h5py.Dataset)
    '''
    def print_name_and_shape(name, obj):
        # check if the object is a dataset
        if isinstance(obj, h5py.Dataset):
            try:
                units = obj.attrs["units"]
            except KeyError:
                units = 'No units specified'
            
            print("Name:", name, ", shape:", obj.shape, ", units:", units)

    f.visititems(print_name_and_shape)

#====================================================================
#====================================================================
#====================================================================
#====================================================================

def ExtractGRIBIris(path, var_names='all', 
                    sclice_over_var='model_level_number', 
                    print_keys=False, print_sum=False, num_examples=1,
                    use_dask_array=False, 
                    essential_var_names=['forecast_reference_time', 'model_level_number', 'latitude', 'longitude']):
    '''
    This will read the desired variables from an grib file.
    
    Returns: A dict of dask arrays or numpy arrays of the variables desired
    
    Parameters:
    - path (str) - Path to the grib file
    - var_names (list, optional) - The names of the variable to be extracted.
    - slice_over_var (str, optional) - variable to slice over when extracting the data from the iris cube
    - print_keys (Bool, optional) - Print the names of the variables an iris cube
    - print_sum (Bool, optional) - Print out a summary of the dataset
    - num_examples (int, optional) - The number of cubes to print
    - use_dask_array (Bool, optional) - If True, stores the contents of the grib file in a Dask array,
      if false, stores it as a numpy array
    '''
    assert path.endswith('.grib'), "File must be .grib. Got:\n"+path
    #----------------------------------------------------------------
    cubes = iris_grib.load_cubes(path)
    cubes = list(cubes)
    print("type(cube[0]):",type(cubes[0]))
    #----------------------------------------------------------------
    ''' Check all cubes have the same shape '''
    for i in range(len(cubes) - 1):
        # Get the shape of the current cube and the next cube
        shape1 = cubes[i].shape
        shape2 = cubes[i + 1].shape
        # If the shapes are not equal, print the cubes
        if not (shape1 == shape2):
            print(f"Shape mismatch between cubes {i} and {i + 1}")
            print(f"Cube {i}: {cubes[i]}")
            print(f"Cube {i + 1}: {cubes[i + 1]}")
            return # Exit function
    #----------------------------------------------------------------
    if print_keys:
        print(vars(cubes[0]))
    #----------------------------------------------------------------
    unique_vars, unique_vars_idx, unique_dates, unique_levels = GetUniqueGRIBIris(cubes)
    #----------------------------------------------------------------
    if not ((var_names == 'all') or (type(var_names) == list)):
        var_names = [var_names]
    #----------------------------------------------------------------
    if (var_names == 'all'):
        var_names = unique_vars
    #----------------------------------------------------------------
    valid_var_names_str = ''
    for name in unique_vars:
        valid_var_names_str += name + ', '

    if not (var_names == 'all'):
        assert all(var_name in unique_vars for var_name in var_names), "Invalid variable name. Valid variables:\n"+valid_var_names_str
    #----------------------------------------------------------------
    data_dim = cubes[0].data.shape # Shouldn't matter which cube is used
    big_data_shape = (len(unique_dates),) + (len(unique_levels),) + data_dim
    #----------------------------------------------------------------
    if print_sum:
        PrintSumGRIBIris(path, cubes, unique_vars, unique_vars_idx, unique_dates, unique_levels, num_examples)
    #----------------------------------------------------------------
    var_dict = {'forecast_reference_time': unique_dates, 
                'model_level_number': unique_levels}
    #----------------------------------------------------------------
    if use_dask_array:
        data_arr = da.empty(shape=big_data_shape, dtype=float)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        var_dict['latitude'] = da.from_array(cubes[0].coord('latitude').points)
        var_dict['longitude'] = da.from_array(cubes[0].coord('longitude').points)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        for var_name in var_names:
            var_dict[var_name] = da.empty(shape=(big_data_shape), dtype=float)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        for cube in cubes:
            s = 0
            for subcube in cube.slices_over(sclice_over_var):
                #print(cube.standard_name, name_idx, subcube.coord('model_level_number').points[0], level_idx)
                level_idx = unique_levels.index(subcube.coord('model_level_number').points[0])

                coord_str = str(subcube.coord('forecast_reference_time'))
                date_str = FindDate(coord_str, 'points')
                date_idx = unique_dates.index(date_str)

                #print(cube.standard_name, name_idx, subcube.coord('model_level_number').points[0], level_idx, date_str, date_idx, ", subcube:", s, ", subname:", subcube.standard_name)

                var_dict[subcube.standard_name][date_idx, level_idx, ...] = subcube.data
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    else:
        data_arr = np.empty(shape=big_data_shape, dtype=float)
        print(data_arr.shape)

        var_dict['latitude'] = cubes[0].coord('latitude').points
        var_dict['longitude'] = cubes[0].coord('longitude').points
         # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        for var_name in var_names:
            var_dict[var_name] = np.empty(shape=(big_data_shape), dtype=float)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        for cube in cubes:
            if (cube.standard_name in var_names):
                s = 0
                for subcube in cube.slices_over('model_level_number'):
                    #print(cube.standard_name, name_idx, subcube.coord('model_level_number').points[0], level_idx)
                    level_idx = unique_levels.index(subcube.coord('model_level_number').points[0])

                    coord_str = str(subcube.coord('forecast_reference_time'))
                    date_str = FindDate(coord_str, 'points')
                    date_idx = unique_dates.index(date_str)

                    #print(cube.standard_name, subcube.coord('model_level_number').points[0], level_idx, date_str, date_idx, ", subcube:", s, ", subname:", subcube.standard_name)

                    var_dict[subcube.standard_name][date_idx, level_idx, ...] = subcube.data
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    return var_dict

def PrintSumGRIBIris(path, cubes, unique_vars, unique_vars_idx, unique_dates, unique_levels, num_cubes_to_print):
    print("+++++++++++++++++ Summary +++++++++++++++++")
    
    print("Number of cubes:", len(cubes))
    print("Variables:", unique_vars)
    print("Date range:", unique_dates[0], "to",unique_dates[-1])
    print("Levels:", unique_levels)
    print("Min lat.:", round(min(cubes[0].coord('latitude').points), 3), ", max. lat.:", round(max(cubes[0].coord('latitude').points), 3))
    print("Min lon.:", round(min(cubes[0].coord('longitude').points), 3), ", max. lon.:", round(max(cubes[0].coord('longitude').points), 3))
    
    ds = gdal.Open(path)
    crs = ds.GetProjectionRef() # get the CRS as a string
    print("crs:", crs)
    print(" - - - - - - - - - Data - - - - - - - - - -")
    for i in range(len(unique_vars_idx)):
        print(" - name:", cubes[unique_vars_idx[i]].standard_name, ", units:", cubes[unique_vars_idx[i]].units, ", shape:", cubes[unique_vars_idx[i]].data.shape)
        print(" - + Spatial-temporal info:")
        for j in range(len(cubes[unique_vars_idx[i]]._dim_coords_and_dims)):
            print(" - + -", cubes[unique_vars_idx[i]]._dim_coords_and_dims[j])
        for k in range(len(cubes[unique_vars_idx[i]]._aux_coords_and_dims)):
            print(" - + -", cubes[unique_vars_idx[i]]._aux_coords_and_dims[k])
        print(" - - - - - -")

    print(" - + - + - Example cubes - + - + - ")
    for i in range(num_cubes_to_print):
        print("     ----- Cube", i, "-----")
        print(cubes[i])
    print("+++++++++++++++++++++++++++++++++++++++++++")

def GetUniqueGRIBIris(cubes):
    unique_vars = []
    unique_vars_idx = []
    unique_dates = []
    unique_levels = []

    for i in range(len(cubes)):
        if not (cubes[i].standard_name in unique_vars):
            unique_vars.append(cubes[i].standard_name)
            unique_vars_idx.append(i)
        
        coord_str = str(cubes[i].coord('forecast_reference_time'))
        date_str = FindDate(coord_str, 'points')
        if not (date_str in unique_dates):
            unique_dates.append(date_str)

        if not(cubes[i].coord('model_level_number').points[0] in unique_levels):
            unique_levels.append(cubes[i].coord('model_level_number').points[0])

    unique_dates.sort()

    return unique_vars, unique_vars_idx, unique_dates, unique_levels









































# def PrintSumGRIB(grbs_list, grbs_pointer, unique_dict, data_names, num_to_display=1, 
#                  vars_to_examine=['parameterName', 'level', 'dataDate',]):
#     '''
#     Prints a summary of the .grib dataset. It will also extract the unique values of the variables
#     specified in vars_to_examine.

#     Returns the unique values of the variables specified in vars_to_examine.

#     Parameters
#      - grbs_list (list): List of grib messages
#      - grbs_pointer (??): Pointer to the grib file
#      - data_names (list): List of the variable names you can use in grb.select(name='my_var_name')
#      - num_to_display (int, optional): How many grib messages to display
#      - vars_to_examine (list, optional): Which variables you want to look at
#     '''
#     for key in unique_dict:
#         print(key, unique_dict[key])
#     print("Var names for select:", data_names)

#     print(" - + - + - Examples - + - + -")
#     for i in range(num_to_display):
#         print(grbs_list[i])
#         for name in data_names:
#             #print(" ||||||| name:", name)
#             temp = grbs_pointer.select(name=name)[0]
#             temp_lat, temp_lon = temp.latlons()
#             print("  values:", np.shape(temp.values), '||| lat:', np.shape(temp_lat), ", range:", min(temp_lat.ravel()), "-", max(temp_lat.ravel()), "||| lon:", np.shape(temp_lon), ", range:", min(temp_lon.ravel()), "-", max(temp_lon.ravel()))
#         print("    - - -")
#     print("===================================================")

# def GetUniqueGRIB(grbs_list, grbs_pointer, var_names):
#     '''
#      - Grib open object
#     '''
#     grb = grbs_list[0] # We are assuming each item in the grbs_list has the same attributes
#     valid_keys_str = ''
#     for key in list(grb.keys()):
#         valid_keys_str += key + ','

#     assert all(var_name in list(grb.keys()) for var_name in var_names), "Some variable names are not attributes. Valid attributes are:\n"+ valid_keys_str

#     data_names = []   
#     unique_dict = {var_name:[] for var_name in var_names}
#     for grb in grbs_list:

#         substrings = str(grb).split(':')
#         if not (substrings[1] in data_names):
#             data_names.append(substrings[1]) # Pull the variable's name from the grib message

#         for var_name in var_names:
#             lol = unique_dict[var_name]
            
#             if not (grb[var_name] in lol):
#                 lol.append(grb[var_name])
            
#             unique_dict[var_name] = lol

#     return unique_dict, data_names

# def ExtractGRIB(path, var_names='all', 
#                 essential_parameter_names=['level', 'dataDate', 'parameterName'], 
#                 print_keys=True, print_sum=False, num_examples=3, use_dask_array=False):
#     '''
#     This will read the desired variables from an hdf5 file.
    
#     Returns: A dict of dask arrays or numpy arrays of the variables desired
    
#     Parameters:
#     - path (str) - Path to the grib file
#     - var_names (list, optional) - The names of the variable to be extracted. The names of these variables are printed
#       under 'var names for select' when print_sum is True. If 'all', will return all of the valid variables
#     - essential_parameter_names (list, optional) - The names of 'essential' parameters, e.g. the date of collection, the level
#       of the model the data is at. All valid essential_parameter_names are printed when print_keys=True
#     - print_keys (Bool, optional) - Print the keys of the first grib message
#     - print_sum (Bool, optional) - Print out a summary of the dataset
#     - num_examples (int, optional) - The number of grib messages to print
#     - use_dask_array (Bool, optional) - If True, stores the contents of the grib file in a Dask array,
#       if false, stores it as a numpy array
#     '''
#     assert path.endswith('.grib'), "File must be .grib. Got:\n"+path
#     #----------------------------------------------------------------
#     grbs_pointer = pygrib.open(path)
#     grbs_pointer.seek(0)

#     grbs_list = list(grbs_pointer)

#     grb = grbs_list[0]
#     #----------------------------------------------------------------
#     if print_keys:
#         print(grb.keys())
#     #----------------------------------------------------------------
#     unique_dict, data_names = GetUniqueGRIB(grbs_list, grbs_pointer, essential_parameter_names)
#     #----------------------------------------------------------------
#     if print_sum:
#         PrintSumGRIB(grbs_list, grbs_pointer, unique_dict, data_names, num_examples, essential_parameter_names)
#     #----------------------------------------------------------------
    
#     #----------------------------------------------------------------

#     #----------------------------------------------------------------

#     # u = grbs.select(name='U component of wind')[0]
#     # u_lat, u_lon = u.latlons()
#     # print("||||| len", len(grbs.select(name='U component of wind')))
#     # print("||||| u-lat=", np.shape(u_lat))
#     # print("||||| u-lon=", np.shape(u_lon))
#     # print("||||| u.values=", np.shape(u.values))
#     #----------------------------------------------------------------
#     var_dict = {var_name:[] for var_name in var_names}
#     #----------------------------------------------------------------


#     #----------------------------------------------------------------
#     #valid_keys_list, valid_keys_str = GetKeysGRIB(grbs)
    
#     #assert all(var_name in valid_keys_list for var_name in var_names), "Some variable names are not in the file. Valid variable names are:\n"+ valid_keys_str
    
#     #----------------------------------------------------------------
#     grbs_pointer.close()
#     return var_dict




