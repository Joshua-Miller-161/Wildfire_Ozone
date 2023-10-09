import netCDF4 as nc
import h5py
import numpy as np
import xarray as xr
import dask.array as da


def Extract_netCDF4(path, var_names, groups=None, print_sum=False):
    '''
    This will read the desired variables from a .nc file.
    
    Returns: A dict of dask arrays or numpy arrays of the variables desired
    
    Parameters:
    - path (str) - path to the nc file
    - var_names (list) - the names of the variable to be extracted
    - group (list, optional) - list of the name(s) of the group in the hdf5 file to be accessed
    - print_sum (Bool, optional) - print out a summary of the dataset
    '''
    assert path.endswith('.nc') or path.endswith('.nc4'), "File must be .nc"
    #----------------------------------------------------------------
    if not groups == None:
        if not type(groups) == list:
            print('AHHHHHHHHHH', type(groups))
            groups = [groups]
    #----------------------------------------------------------------
    var_dict = {}
    #----------------------------------------------------------------
    ds = nc.Dataset(path, "r", format='NETCDF4')
    #----------------------------------------------------------------
    if groups==None:
        valid_keys = GetKeysNC(ds)

        assert all(var_name in ds.variables.keys() for var_name in var_names), "Some variable names are not in the file. Valid variable names are:\n"+ valid_keys
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        for var_name in var_names:
            var = ds.variables[var_name] # access the variable as a dataset object
            var_dict[var_name] = var[:]  # convert to numpy array using [:]
    #----------------------------------------------------------------
    else:
        for group in groups:
            group_ds = ds.groups[group]

            if print_sum:
                PrintSumNC(group_ds)
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            valid_keys = GetKeysNC(group_ds)

            assert all(var_name in list(group_ds.variables.keys()) for var_name in var_names), "Some variable names are not in the group, "+group+". Valid variable names in this group are:\n"+ valid_keys
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            for var_name in var_names:
                if var_name in list(group_ds.variables.keys()):
                    var = group_ds.variables[var_name]
                    var_dict[var_name] = var[:]
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ds.close() # close the file
    return var_dict



def ExtractHDF5(path, var_names, groups=None, print_sum=False, chunks='auto', to_numpy=True):
    '''
    This will read the desired variables from an hdf5 file.
    
    Returns: A dict of dask arrays or numpy arrays of the variables desired
    
    Parameters:
    - path (str) - path to the hdf5 file
    - var_names (list) - the names of the variable to be extracted
    - group (list, optional) - list of the name(s) of the group in the hdf5 file to be accessed
    - print_sum (Bool, optional) - print out a summary of the dataset
    - chunks (tuple, optional) - the chunk size(s) for from_array(ds["var_name"].values, chunks=)
    - to_numpy (Bool, optional) - if True, returns variable as a numpy array; if False, leaves it as a dask array
    '''
    assert path.endswith('.hdf5'), "File must be .hdf5"
    #----------------------------------------------------------------
    if not groups == None:
        if not type(groups) == list:
            print('AHHHHHHHHHH', type(groups))
            groups = [groups]
    #----------------------------------------------------------------
    var_dict = {}
    #----------------------------------------------------------------
    f = h5py.File(path, "r") # open the file in read mode
    #----------------------------------------------------------------
    if print_sum:
        PrintSumHDF5(f)
    #----------------------------------------------------------------
    valid_keys_list, valid_keys_str = GetKeysHDF5(f)

    assert all(var_name in valid_keys_list for var_name in var_names), "Some variable names are not in the file. Valid variable names are:\n"+ valid_keys_str
    #----------------------------------------------------------------
    if groups==None:
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        for var_name in var_names:
            var = f[var_name] # access the variable as a dataset object
            var_dict[var_name] = var[()] # convert to numpy array using [()]
    #----------------------------------------------------------------
    else:
        for group in groups:
            for var_name in var_names:
                if var_name in list(f[group].keys()):
                    var = f[group+'/'+var_name]
                    lol = var[()]
                    print('TYPE --------- ', type(var), type(lol))
                    var_dict[var_name] = var[()]
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    f.close() # close the file
    print('===== Closed file')
    return var_dict



def PrintSumHDF5(f):
    '''
    Prints the variable name, shape
     - f (h5py.Dataset)
    '''
    def print_name_and_shape(name, obj):
        # check if the object is a dataset
        if isinstance(obj, h5py.Dataset):
            # print its name and shape
            print(name, obj.shape)

        f.visititems(print_name_and_shape)



def PrintSumNC(ds):
    print("===================================================")
    print("SUMMARY :", ds)
    print("===================================================")
    for name, var in ds.variables.items(): # loop through the variables
        print(name, var.shape)



def GetKeysHDF5(f):
    '''
    
     - f (h5py.Dataset)
    '''
    valid_keys_list = []
    valid_keys_str = ''

    for key in f.keys():
        print("+++++++++++", type(f[key]), "++++++++++++")
        if isinstance(f[key], h5py._hl.group.Group): # This 'key' is a group name
            for var in f[key].keys():  # Now iterating through variable names
                valid_keys_list.append(var)
                valid_keys_str += var + ', '
        else:
            valid_keys_list.append(key)
            valid_keys_str += key + ', '

    return valid_keys_list, valid_keys_str



def GetKeysNC(ds):
    valid_keys = ''
    for key in ds.variables.keys():
        valid_keys += key + ', '
    return valid_keys