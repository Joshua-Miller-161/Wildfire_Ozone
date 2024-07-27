import sys
sys.dont_write_bytecode
import numpy as np
import dask.array as da
import iris_grib
from osgeo import gdal
import os

sys.path.append(os.getcwd())
from misc.misc_utils import FindDate
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
    unique_vars, unique_vars_idx, unique_dates, unique_levels, unique_pressures = GetUniqueGRIBIris(cubes)
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
        PrintSumGRIBIris(path, cubes, unique_vars, unique_vars_idx, unique_dates, unique_levels, unique_pressures, num_examples)
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
    for key in var_dict.keys():
        try:
            var_dict[key] = np.squeeze(var_dict[key])
        except ValueError:
            print('lol')
    return var_dict

def PrintSumGRIBIris(path, cubes, unique_vars, unique_vars_idx, unique_dates, unique_levels, unique_pressures, num_cubes_to_print):
    print("+++++++++++++++++ Summary +++++++++++++++++")
    
    print("Number of cubes:", len(cubes))
    print("Variables:", unique_vars)
    print("Date range:", unique_dates[0], "to",unique_dates[-1])
    print("Levels:", unique_levels)
    print("Level pressures:", unique_pressures)
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
    unique_pressures = []

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
            unique_pressures.append(cubes[i].coord('level_pressure').points[0])

    unique_dates.sort()

    return unique_vars, unique_vars_idx, unique_dates, unique_levels, unique_pressures