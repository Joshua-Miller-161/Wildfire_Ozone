import sys
sys.dont_write_bytecode
import netCDF4 as nc
import h5py
import numpy as np
import dask.array as da
#import iris_grib
from osgeo import gdal
import rasterio
import csv
import os


sys.path.append(os.getcwd())
from misc.misc_utils import FindDate

#====================================================================
#====================================================================
#====================================================================
#====================================================================

def Extract_netCDF4(path, var_names, groups=None, print_sum=False, attrs_to_find=None):
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
    assert path.endswith('.nc') or path.endswith('.nc4') or path.endswith('.nc4'), "File must be .nc\Got:"+path
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

    if print_sum:
        print("____________________________")
        print("Valid keys:", valid_keys_list)
        print("____________________________")

    assert all(var_name in valid_keys_list for var_name in var_names), "Some variable names are not in the file. Valid variable names in this are:\n"+ valid_keys
    #----------------------------------------------------------------
    if ((groups==None) or (list(ds.groups.keys())==[])):
        if print_sum:
            PrintSumNC(ds)
        
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
    #----------------------------------------------------------------
    
    ds.close() # close the file
    return var_dict

def GetKeysNC(ds):
    '''
     - ds (nc.Dataset)
    '''
    valid_keys_list = []
    valid_keys_str = ''

    if (len(list(ds.groups.keys())) > 0): # Iterate through the groups
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
        try:
            print(name, var.shape, var.units)
        except AttributeError:
            print(name, var.shape, ", no units provided.")
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
    #assert path.endswith('.hdf5') or path.endswith('.h5') or path.endswith('.hdf'), "File must be .hdf5 or .h5. Got:\n"+path
    #----------------------------------------------------------------
    if print_sum:
        PrintSumHDF5(path)
    #----------------------------------------------------------------
    f = h5py.File(path, "r") # open the file in read mode
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
    valid_keys_list, valid_keys_str = GetKeysHDF5(f)
    
    assert all(var_name in valid_keys_list for var_name in var_names), "Some variable names are not in the file. Valid variable names are:\n"+ valid_keys_str
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

def PrintSumHDF5(path, f=None):
    '''
    Prints the variable name, shape
     - path (str): Path to the hdf5 file. Used to get crs info
     - f (h5py.Dataset): The hdf5 dataset object corresponding to the hdf5 file in path
    '''
    if (f == None):
        f = h5py.File(path, "r")

    print("TYPE:", type(f))
    #----------------------------------------------------------------
    def print_name_and_shape(name, obj):
        if isinstance(obj, h5py.Dataset): # check if the object is a dataset
            units = obj.attrs.get('units', 'No units provided') # get the units attribute or 'unknown' if not found
            try: # try to get the max and min of the dataset
                max = np.max(obj) # get the maximum value of the dataset
                min = np.min(obj) # get the minimum value of the dataset
                print(f'{name}: shape={obj.shape}, units={units}, max={max}, min={min}') # print the name, shape, units, max and min
            except: # if an error occurs, print a message
                print(f'{name}: shape={obj.shape}, units={units}, max and min not available') # print the name, shape and units, and indicate that max and min are not available
                
        else: # otherwise, assume the object is a group
            units = obj.attrs.get('units', 'No units provided') # get the units attribute or 'unknown' if not found
            print(f'{name}: shape=(), units={units}') # print the name, units and empty shape

    if isinstance(f, h5py.Group):
        f.visititems(print_name_and_shape)
    else:
        print("f is type:", type(f))
    #----------------------------------------------------------------
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
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
    print("========================================================")
    f.close()

def print_dataset_summary(hdf5_path):
    """
    Prints the summary of each dataset in the HDF5 file including shape, min, max, and units.

    Parameters:
    - hdf5_path (str): Path to the HDF5 file.
    """
    with h5py.File(hdf5_path, 'r') as file:
        def visit_datasets(name, node):
            if isinstance(node, h5py.Dataset):
                units = node.attrs.get('units', 'No units provided')
                data = node[...]  # Load the data into memory
                print(f'Dataset: {name}')
                print(f'  Shape: {node.shape}')
                print(f'  Units: {units}')
                try:
                    print(f'  Min: {np.min(data)}')
                    print(f'  Max: {np.max(data)}')
                except TypeError:
                    print('  Min/Max: Not available for non-numeric data')
        
        file.visititems(visit_datasets)
#====================================================================
#====================================================================
#====================================================================
#====================================================================

# def ExtractGRIBIris(path, var_names='all', 
#                     sclice_over_var='model_level_number', 
#                     print_keys=False, print_sum=False, num_examples=1,
#                     use_dask_array=False, 
#                     essential_var_names=['forecast_reference_time', 'model_level_number', 'latitude', 'longitude']):
#     '''
#     This will read the desired variables from an grib file.
    
#     Returns: A dict of dask arrays or numpy arrays of the variables desired
    
#     Parameters:
#     - path (str) - Path to the grib file
#     - var_names (list, optional) - The names of the variable to be extracted.
#     - slice_over_var (str, optional) - variable to slice over when extracting the data from the iris cube
#     - print_keys (Bool, optional) - Print the names of the variables an iris cube
#     - print_sum (Bool, optional) - Print out a summary of the dataset
#     - num_examples (int, optional) - The number of cubes to print
#     - use_dask_array (Bool, optional) - If True, stores the contents of the grib file in a Dask array,
#       if false, stores it as a numpy array
#     '''
#     assert path.endswith('.grib'), "File must be .grib. Got:\n"+path
#     #----------------------------------------------------------------
#     cubes = iris_grib.load_cubes(path)
#     cubes = list(cubes)
#     print("type(cube[0]):",type(cubes[0]))
#     #----------------------------------------------------------------
#     ''' Check all cubes have the same shape '''
#     for i in range(len(cubes) - 1):
#         # Get the shape of the current cube and the next cube
#         shape1 = cubes[i].shape
#         shape2 = cubes[i + 1].shape
#         # If the shapes are not equal, print the cubes
#         if not (shape1 == shape2):
#             print(f"Shape mismatch between cubes {i} and {i + 1}")
#             print(f"Cube {i}: {cubes[i]}")
#             print(f"Cube {i + 1}: {cubes[i + 1]}")
#             return # Exit function
#     #----------------------------------------------------------------
#     if print_keys:
#         print(vars(cubes[0]))
#     #----------------------------------------------------------------
#     unique_vars, unique_vars_idx, unique_dates, unique_levels, unique_pressures = GetUniqueGRIBIris(cubes)
#     #----------------------------------------------------------------
#     if not ((var_names == 'all') or (type(var_names) == list)):
#         var_names = [var_names]
#     #----------------------------------------------------------------
#     if (var_names == 'all'):
#         var_names = unique_vars
#     #----------------------------------------------------------------
#     valid_var_names_str = ''
#     for name in unique_vars:
#         valid_var_names_str += name + ', '

#     if not (var_names == 'all'):
#         assert all(var_name in unique_vars for var_name in var_names), "Invalid variable name. Valid variables:\n"+valid_var_names_str
#     #----------------------------------------------------------------
#     data_dim = cubes[0].data.shape # Shouldn't matter which cube is used
#     big_data_shape = (len(unique_dates),) + (len(unique_levels),) + data_dim
#     #----------------------------------------------------------------
#     if print_sum:
#         PrintSumGRIBIris(path, cubes, unique_vars, unique_vars_idx, unique_dates, unique_levels, unique_pressures, num_examples)
#     #----------------------------------------------------------------
#     var_dict = {'forecast_reference_time': unique_dates, 
#                 'model_level_number': unique_levels}
#     #----------------------------------------------------------------
#     if use_dask_array:
#         data_arr = da.empty(shape=big_data_shape, dtype=float)
#         # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#         var_dict['latitude'] = da.from_array(cubes[0].coord('latitude').points)
#         var_dict['longitude'] = da.from_array(cubes[0].coord('longitude').points)
#         # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#         for var_name in var_names:
#             var_dict[var_name] = da.empty(shape=(big_data_shape), dtype=float)
#         # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#         for cube in cubes:
#             s = 0
#             for subcube in cube.slices_over(sclice_over_var):
#                 #print(cube.standard_name, name_idx, subcube.coord('model_level_number').points[0], level_idx)
#                 level_idx = unique_levels.index(subcube.coord('model_level_number').points[0])

#                 coord_str = str(subcube.coord('forecast_reference_time'))
#                 date_str = FindDate(coord_str, 'points')
#                 date_idx = unique_dates.index(date_str)

#                 #print(cube.standard_name, name_idx, subcube.coord('model_level_number').points[0], level_idx, date_str, date_idx, ", subcube:", s, ", subname:", subcube.standard_name)

#                 var_dict[subcube.standard_name][date_idx, level_idx, ...] = subcube.data
#         # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#     else:
#         data_arr = np.empty(shape=big_data_shape, dtype=float)
#         print(data_arr.shape)

#         var_dict['latitude'] = cubes[0].coord('latitude').points
#         var_dict['longitude'] = cubes[0].coord('longitude').points
#          # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#         for var_name in var_names:
#             var_dict[var_name] = np.empty(shape=(big_data_shape), dtype=float)
#         # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#         for cube in cubes:
#             if (cube.standard_name in var_names):
#                 s = 0
#                 for subcube in cube.slices_over('model_level_number'):
#                     #print(cube.standard_name, name_idx, subcube.coord('model_level_number').points[0], level_idx)
#                     level_idx = unique_levels.index(subcube.coord('model_level_number').points[0])

#                     coord_str = str(subcube.coord('forecast_reference_time'))
#                     date_str = FindDate(coord_str, 'points')
#                     date_idx = unique_dates.index(date_str)

#                     #print(cube.standard_name, subcube.coord('model_level_number').points[0], level_idx, date_str, date_idx, ", subcube:", s, ", subname:", subcube.standard_name)

#                     var_dict[subcube.standard_name][date_idx, level_idx, ...] = subcube.data
#         # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#     for key in var_dict.keys():
#         try:
#             var_dict[key] = np.squeeze(var_dict[key])
#         except ValueError:
#             print('lol')
#     return var_dict

# def PrintSumGRIBIris(path, cubes, unique_vars, unique_vars_idx, unique_dates, unique_levels, unique_pressures, num_cubes_to_print):
#     print("+++++++++++++++++ Summary +++++++++++++++++")
    
#     print("Number of cubes:", len(cubes))
#     print("Variables:", unique_vars)
#     print("Date range:", unique_dates[0], "to",unique_dates[-1])
#     print("Levels:", unique_levels)
#     print("Level pressures:", unique_pressures)
#     print("Min lat.:", round(min(cubes[0].coord('latitude').points), 3), ", max. lat.:", round(max(cubes[0].coord('latitude').points), 3))
#     print("Min lon.:", round(min(cubes[0].coord('longitude').points), 3), ", max. lon.:", round(max(cubes[0].coord('longitude').points), 3))
    
#     ds = gdal.Open(path)
#     crs = ds.GetProjectionRef() # get the CRS as a string
#     print("crs:", crs)
#     print(" - - - - - - - - - Data - - - - - - - - - -")
#     for i in range(len(unique_vars_idx)):
#         print(" - name:", cubes[unique_vars_idx[i]].standard_name, ", units:", cubes[unique_vars_idx[i]].units, ", shape:", cubes[unique_vars_idx[i]].data.shape)
#         print(" - + Spatial-temporal info:")
#         for j in range(len(cubes[unique_vars_idx[i]]._dim_coords_and_dims)):
#             print(" - + -", cubes[unique_vars_idx[i]]._dim_coords_and_dims[j])
#         for k in range(len(cubes[unique_vars_idx[i]]._aux_coords_and_dims)):
#             print(" - + -", cubes[unique_vars_idx[i]]._aux_coords_and_dims[k])
#         print(" - - - - - -")

#     print(" - + - + - Example cubes - + - + - ")
#     for i in range(num_cubes_to_print):
#         print("     ----- Cube", i, "-----")
#         print(cubes[i])
#     print("+++++++++++++++++++++++++++++++++++++++++++")

# def GetUniqueGRIBIris(cubes):
#     unique_vars = []
#     unique_vars_idx = []
#     unique_dates = []
#     unique_levels = []
#     unique_pressures = []

#     for i in range(len(cubes)):
#         if not (cubes[i].standard_name in unique_vars):
#             unique_vars.append(cubes[i].standard_name)
#             unique_vars_idx.append(i)
        
#         coord_str = str(cubes[i].coord('forecast_reference_time'))
#         date_str = FindDate(coord_str, 'points')
#         if not (date_str in unique_dates):
#             unique_dates.append(date_str)

#         if not(cubes[i].coord('model_level_number').points[0] in unique_levels):
#             unique_levels.append(cubes[i].coord('model_level_number').points[0])
#             unique_pressures.append(cubes[i].coord('level_pressure').points[0])

#     unique_dates.sort()

#     return unique_vars, unique_vars_idx, unique_dates, unique_levels, unique_pressures

#====================================================================
#====================================================================
#====================================================================
#====================================================================

def SplitDataFrame(df, column, save_new_files=False, new_files_folder=None):
    '''
    Creates subsets of df where all of the entries in 'column' are the same

    Returns a dictionary of subsets of the original dataframe

    - df (Pandas dataframe): The dataframe to split
    - column (str): The column in which the unique values will be searched
    - save_new_file (Bool, optional): Whether to save the new dataframes as csv files
    - new_files_folder (str, optional): Path to folder to save the subsets
    '''
    #----------------------------------------------------------------
    cols_str = ''
    for col in df.columns:
        cols_str += col +', '
    assert column in df.columns, column+" is not a valid column name. Valid columns:\n"+cols_str
    #----------------------------------------------------------------
    sub_dfs_dict = {}

    for val in df[column].unique():
        print(">> Searching for", val)
        sub_dfs_dict[val] = df[df[column] == val]
        print(">> Finished with", val, "\n  - - - -")
    #----------------------------------------------------------------
    if save_new_files:
        for key in sub_dfs_dict.keys():
            sub_dfs_dict[key].to_csv(os.path.join(new_files_folder, key+'.csv'), index=False)
            print(">> Saved", key+'.csv')
    #----------------------------------------------------------------
    return sub_dfs_dict