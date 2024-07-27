import sys
sys.dont_write_bytecode
import h5py
import numpy as np
from osgeo import gdal
import rasterio
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
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