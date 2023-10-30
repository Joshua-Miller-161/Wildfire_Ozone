import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
from shapely.geometry import Point
import netCDF4 as nc
import numpy as np
import pandas as pd
import re
import sys
import os

sys.path.append(os.getcwd())
from preprocessing_funcs import Scale
from extraction_funcs import ExtractHDF5
from misc.misc_utils import FindDate
#====================================================================
''' Get data '''
path1 = "/Users/joshuamiller/Documents/Lancaster/Data/Fire/DL_FIRE_M-C61_396339/fire_archive_M-C61_396339.csv"
df = pd.read_csv(path1)
print(df)
#====================================================================
fig, ax = plt.subplots(1,1,figsize=(9,7))

#====================================================================
''' World map '''
world = gpd.read_file("/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")
#====================================================================
''' Plot fire '''

world.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map

plt.show()





















# # print("IRIS:", len(cubes), type(cubes[0])) 
# # print("===========================================================")
# # print("===========================================================")
# # print("vars(cubes[0]):", vars(cubes[0]))
# # print("===========================================================")
# # print("===========================================================")
# # print("cubes[0].__dir__:", cubes[0].__dir__())
# # print("===========================================================")
# # print("===========================================================")
# # print("cubes[0].__dict__:", cubes[0].__dict__)
# # print("===========================================================")
# # print("===========================================================")
# # print("cubes[0].data.__dir__():", cubes[0].units, cubes[0].data.__dir__())
# # print("===========================================================")
# # print("===========================================================")
# # print("cubes[0].coord.__dir__():", cubes[0].coord.__dir__())
# # print("===========================================================")
# # print("===========================================================")
# # cube_num = 4
# # print("Data:", cubes[cube_num].standard_name, cubes[cube_num].data.shape, cubes[cube_num].data.shape, type(cubes[cube_num].data), cubes[cube_num].data)
# # print("===========================================================")
# # print(cubes[cube_num]._dim_coords_and_dims[0][0], type(cubes[cube_num]._dim_coords_and_dims[0][0]))
# # print("-")
# # print(cubes[cube_num]._dim_coords_and_dims[1][0], type(cubes[cube_num]._dim_coords_and_dims[1][0]))
# # print("-")
# # print(cubes[cube_num]._dim_coords_and_dims[0][1], type(cubes[cube_num]._dim_coords_and_dims[0][1]))
# # print("-")
# # print(cubes[cube_num]._dim_coords_and_dims[1][1], type(cubes[cube_num]._dim_coords_and_dims[1][1]))
# # print("===========================================================")
# # print(cubes[cube_num].coord('latitude').points, type(cubes[cube_num].coord('latitude').points))
# # print(cubes[cube_num].coord('longitude').points - 360, type(cubes[cube_num].coord('longitude').points))
# # print(cubes[cube_num].coord('model_level_number').points, type(cubes[cube_num].coord('model_level_number').points))
# # print(cubes[cube_num].coord('time').points, type(cubes[cube_num].coord('time').points))
# # print(cubes[cube_num].coord('forecast_reference_time'), cubes[cube_num].coord('forecast_reference_time').points)
# # print("===========================================================")
# # print("===========================================================")