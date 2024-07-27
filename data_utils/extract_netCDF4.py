import sys
sys.dont_write_bytecode
import netCDF4 as nc
from osgeo import gdal
import rasterio
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

print(">> netCDF4 version =", nc.__version__)
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