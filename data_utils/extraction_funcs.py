import netCDF4 as nc
import h5py
import numpy as np
import xarray as xr
import dask.array as da
import pygrib
import iris_grib
import csv
from pprint import pprint

def Extract_netCDF4(path, var_names, groups=None, print_sum=False):
    '''
    This will read the desired variables from a .nc file.
    
    Returns: A dict of dask arrays or numpy arrays of the variables desired
    
    Parameters:
    - path (str) - path to the nc file
    - var_names (list) - the names of the variable(s) to be extracted
    - group (list, optional) - list of the name(s) of the group in the hdf5 file to be accessed.
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
    ds = nc.Dataset(path, "r", format='NETCDF4')
    #----------------------------------------------------------------
    if groups == 'all':
        groups = list(ds.groups.keys())
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


def ExtractGRIB(path, var_names='all', 
                essential_parameter_names=['level', 'dataDate', 'parameterName'], 
                print_keys=True, print_sum=False, num_examples=3, use_dask_array=False):
    '''
    This will read the desired variables from an hdf5 file.
    
    Returns: A dict of dask arrays or numpy arrays of the variables desired
    
    Parameters:
    - path (str) - Path to the grib file
    - var_names (list, optional) - The names of the variable to be extracted. The names of these variables are printed
      under 'var names for select' when print_sum is True. If 'all', will return all of the valid variables
    - essential_parameter_names (list, optional) - The names of 'essential' parameters, e.g. the date of collection, the level
      of the model the data is at. All valid essential_parameter_names are printed when print_keys=True
    - print_keys (Bool, optional) - Print the keys of the first grib message
    - print_sum (Bool, optional) - Print out a summary of the dataset
    - num_examples (int, optional) - The number of grib messages to print
    - use_dask_array (Bool, optional) - If True, stores the contents of the grib file in a Dask array,
      if false, stores it as a numpy array
    '''
    assert path.endswith('.grib'), "File must be .grib. Got:\n"+path
    #----------------------------------------------------------------
    grbs_pointer = pygrib.open(path)
    grbs_pointer.seek(0)

    grbs_list = list(grbs_pointer)

    grb = grbs_list[0]
    #----------------------------------------------------------------
    if print_keys:
        print(grb.keys())
    #----------------------------------------------------------------
    unique_dict, data_names = GetUniqueGRIB(grbs_list, grbs_pointer, essential_parameter_names)
    #----------------------------------------------------------------
    if print_sum:
        PrintSumGRIB(grbs_list, grbs_pointer, unique_dict, data_names, num_examples, essential_parameter_names)
    #----------------------------------------------------------------
    
    #----------------------------------------------------------------

    #----------------------------------------------------------------

    # u = grbs.select(name='U component of wind')[0]
    # u_lat, u_lon = u.latlons()
    # print("||||| len", len(grbs.select(name='U component of wind')))
    # print("||||| u-lat=", np.shape(u_lat))
    # print("||||| u-lon=", np.shape(u_lon))
    # print("||||| u.values=", np.shape(u.values))
    #----------------------------------------------------------------
    var_dict = {var_name:[] for var_name in var_names}
    #----------------------------------------------------------------


    #----------------------------------------------------------------
    #valid_keys_list, valid_keys_str = GetKeysGRIB(grbs)
    
    #assert all(var_name in valid_keys_list for var_name in var_names), "Some variable names are not in the file. Valid variable names are:\n"+ valid_keys_str
    
    #----------------------------------------------------------------
    grbs_pointer.close()
    return var_dict






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


def PrintSumNC(ds):
    print("SUMMARY :", ds)
    print("===================================================")
    for name, var in ds.variables.items(): # loop through the variables
        print(name, var.shape)


def PrintSumGRIB(grbs_list, grbs_pointer, unique_dict, data_names, num_to_display=1, 
                 vars_to_examine=['parameterName', 'level', 'dataDate',]):
    '''
    Prints a summary of the .grib dataset. It will also extract the unique values of the variables
    specified in vars_to_examine.

    Returns the unique values of the variables specified in vars_to_examine.

    Parameters
     - grbs_list (list): List of grib messages
     - grbs_pointer (??): Pointer to the grib file
     - data_names (list): List of the variable names you can use in grb.select(name='my_var_name')
     - num_to_display (int, optional): How many grib messages to display
     - vars_to_examine (list, optional): Which variables you want to look at
    '''
    for key in unique_dict:
        print(key, unique_dict[key])
    print("Var names for select:", data_names)

    print(" - + - + - Examples - + - + -")
    for i in range(num_to_display):
        print(grbs_list[i])
        for name in data_names:
            #print(" ||||||| name:", name)
            temp = grbs_pointer.select(name=name)[0]
            temp_lat, temp_lon = temp.latlons()
            print("  values:", np.shape(temp.values), '||| lat:', np.shape(temp_lat), ", range:", min(temp_lat.ravel()), "-", max(temp_lat.ravel()), "||| lon:", np.shape(temp_lon), ", range:", min(temp_lon.ravel()), "-", max(temp_lon.ravel()))
        print("    - - -")
    print("===================================================")



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



def GetKeysNC(ds):
    '''
     - ds (nc.Dataset)
    '''
    valid_keys_list = []
    valid_keys_str = ''

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

    return valid_keys_list, valid_keys_str


def GetUniqueGRIB(grbs_list, grbs_pointer, var_names):
    '''
     - Grib open object
    '''
    grb = grbs_list[0] # We are assuming each item in the grbs_list has the same attributes
    valid_keys_str = ''
    for key in list(grb.keys()):
        valid_keys_str += key + ','

    assert all(var_name in list(grb.keys()) for var_name in var_names), "Some variable names are not attributes. Valid attributes are:\n"+ valid_keys_str

    data_names = []   
    unique_dict = {var_name:[] for var_name in var_names}
    for grb in grbs_list:

        substrings = str(grb).split(':')
        if not (substrings[1] in data_names):
            data_names.append(substrings[1]) # Pull the variable's name from the grib message

        for var_name in var_names:
            lol = unique_dict[var_name]
            
            if not (grb[var_name] in lol):
                lol.append(grb[var_name])
            
            unique_dict[var_name] = lol

    return unique_dict, data_names



def ExtractGRIB(path, var_names='all', 
                essential_parameter_names=['level', 'dataDate', 'parameterName'], 
                print_keys=True, print_sum=False, num_examples=3, use_dask_array=False):
    '''
    This will read the desired variables from an hdf5 file.
    
    Returns: A dict of dask arrays or numpy arrays of the variables desired
    
    Parameters:
    - path (str) - Path to the grib file
    - var_names (list, optional) - The names of the variable to be extracted. The names of these variables are printed
      under 'var names for select' when print_sum is True. If 'all', will return all of the valid variables
    - essential_parameter_names (list, optional) - The names of 'essential' parameters, e.g. the date of collection, the level
      of the model the data is at. All valid essential_parameter_names are printed when print_keys=True
    - print_keys (Bool, optional) - Print the keys of the first grib message
    - print_sum (Bool, optional) - Print out a summary of the dataset
    - num_examples (int, optional) - The number of grib messages to print
    - use_dask_array (Bool, optional) - If True, stores the contents of the grib file in a Dask array,
      if false, stores it as a numpy array
    '''
    assert path.endswith('.grib'), "File must be .grib. Got:\n"+path
    #----------------------------------------------------------------
    grbs_pointer = pygrib.open(path)
    grbs_pointer.seek(0)

    grbs_list = list(grbs_pointer)

    grb = grbs_list[0]
    #----------------------------------------------------------------
    if print_keys:
        print(grb.keys())
    #----------------------------------------------------------------
    unique_dict, data_names = GetUniqueGRIB(grbs_list, grbs_pointer, essential_parameter_names)
    #----------------------------------------------------------------
    if print_sum:
        PrintSumGRIB(grbs_list, grbs_pointer, unique_dict, data_names, num_examples, essential_parameter_names)
    #----------------------------------------------------------------
    
    #----------------------------------------------------------------

    #----------------------------------------------------------------

    # u = grbs.select(name='U component of wind')[0]
    # u_lat, u_lon = u.latlons()
    # print("||||| len", len(grbs.select(name='U component of wind')))
    # print("||||| u-lat=", np.shape(u_lat))
    # print("||||| u-lon=", np.shape(u_lon))
    # print("||||| u.values=", np.shape(u.values))
    #----------------------------------------------------------------
    var_dict = {var_name:[] for var_name in var_names}
    #----------------------------------------------------------------


    #----------------------------------------------------------------
    #valid_keys_list, valid_keys_str = GetKeysGRIB(grbs)
    
    #assert all(var_name in valid_keys_list for var_name in var_names), "Some variable names are not in the file. Valid variable names are:\n"+ valid_keys_str
    
    #----------------------------------------------------------------
    grbs_pointer.close()
    return var_dict