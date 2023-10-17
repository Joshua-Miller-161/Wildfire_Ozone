import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import argparse
import sys
import os
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime
from dateutil.parser import parse
import yaml
import csv
#====================================================================
sys.path.append('data_utils/extraction_funcs.py')

#====================================================================
from extraction_funcs import ExtractHDF5

#====================================================================
''' Load variables from config file '''
with open('data_utils/GOME_filter_config.yml', 'r') as c:
    config = yaml.load(c, Loader=yaml.FullLoader)

min_lat = config['BOUNDING_BOX']['min_lat']
max_lat = config['BOUNDING_BOX']['max_lat']
min_lon = config['BOUNDING_BOX']['min_lon']
max_lon = config['BOUNDING_BOX']['max_lon']

vars_to_extract = config['VARIABLES']

in_folder = config['FOLDERS']['in_folder']
out_folder = config['FOLDERS']['out_folder']
map = config['FOLDERS']['map']

#====================================================================
''' Get the filenames and remove junk that isn't an hdf5 file '''
file_names = os.listdir(in_folder)
file_names.sort()
for file in file_names:
    if not (file[-5:] == '.hdf5'):
        file_names.remove(file)

#====================================================================
''' Group together files which cover the same days '''
# print(">> Grouping files")
# file_dict = {}

# for i in range(len(file_names)):
#     timestamps = ExtractHDF5(os.path.join(in_folder, file_names[i]), 'Time', groups='all', print_sum=False)
    
#     unique_dates_in_file = []
#     for timestamp in timestamps['Time']:
#         date_obj = parse(timestamp)
#         y_m_d = str(date_obj.year) + '-' + str(date_obj.month) + '-' + str(date_obj.day)

#         if not (y_m_d in unique_dates_in_file):
#             unique_dates_in_file.append(y_m_d)
#             #print("Found unique date:", y_m_d, timestamp, unique_dates_in_file)

#     for date in unique_dates_in_file:
#         try:
#             file_dict[date].append(file_names[i])
#         except:
#             KeyError
#             file_dict[date] = [file_names[i]] # First time finding a file that contains this date

# print(">> Grouped files\n--")

# #====================================================================
# ''' Spatially filter lat/lon values for each day and write to the appropriate file '''
# print('>> Filtering data')
# for date in list(file_dict.keys()):
#     print(">>>> Searching for data on", date)

#     filtered_dict = dict.fromkeys(vars_to_extract, [])

#     lon_and = []
#     lat_and = []
#     time_and = []
#     ozone_and = []

#     curr_year  = int(date.split('-')[0])
#     curr_month = int(date.split('-')[1])
#     curr_day   = int(date.split('-')[2])

#     min_timestamp = datetime(year=curr_year, month=curr_month, day=curr_day, hour=0, minute=0, second=0)
#     max_timestamp = datetime(year=curr_year, month=curr_month, day=curr_day, hour=23, minute=59, second=59)

#     for file in file_dict[date]:
#         dict_ = ExtractHDF5(os.path.join(in_folder, file),
#                             vars_to_extract,
#                             groups='all',
#                             print_sum=False)
#         #print('++++++++++++++++++++++++++++++++++++')
#         #print("dict_=", dict_)
#         #print('++++++++++++++++++++++++++++++++++++')
        
#         count = 0

#         for i in range(np.shape(dict_['Time'])[0]): # Shouldn't matter which var you pick
#             curr_timestamp = parse(dict_['Time'][i])

#             if ((min_timestamp <= curr_timestamp) and (curr_timestamp <= max_timestamp)):
#                 #print("min time:", min_timestamp, ", curr:", curr_timestamp, ", max time:", max_timestamp)
#                 if ((min_lat <= dict_['LatitudeCenter'][i]) and (dict_['LatitudeCenter'][i] <= max_lat)):
#                     #print("min lat:", min_lat, ", curr:", dict_['LatitudeCenter'][i], ", max lat:", max_timestamp)
#                     if ((min_lon <= dict_['LongitudeCenter'][i]) and (dict_['LongitudeCenter'][i] <= max_lon)):
#                         #print("min lon:", min_timestamp, ", curr:", dict_['LongitudeCenter'][i], ", max lon:", max_timestamp)
                        
#                         #print("dict_['LongitudeCenter'][i]=", dict_['LongitudeCenter'][i])
#                         lon_and.append(dict_['LongitudeCenter'][i])
#                         lat_and.append(dict_['LatitudeCenter'][i])
#                         time_and.append(dict_['Time'][i]) # Get rid of 'b' character at the front
#                         ozone_and.append(dict_['IntegratedVerticalProfile'][i])
                        
#                         count += 1

#         filtered_dict['LongitudeCenter'] = lon_and
#         filtered_dict['LatitudeCenter'] = lat_and
#         filtered_dict['Time'] = time_and
#         filtered_dict['IntegratedVerticalProfile'] = ozone_and
#         #print("==========================================")
#         #print("filtered_dict=", filtered_dict)
#         #print("==========================================")
#         print(">>>>>> Found", count, "points in", file)
    
#     with open(os.path.join(out_folder, date+".csv"), "w") as outfile:
#         # pass the csv file to csv.writer function.
#         writer = csv.writer(outfile, dialect='excel')
    
#         # pass the dictionary keys to writerow
#         # function to frame the columns of the csv file
#         writer.writerow(filtered_dict.keys())
    
#         # make use of writerows function to append
#         # the remaining values to the corresponding
#         # columns using zip function.
#         writer.writerows(zip(*filtered_dict.values()))

#     print(">>>> Saved", date, '.csv | Contains', len(filtered_dict['Time']), 'points')

# print(">> Filtered data\n--")

#====================================================================

fig, ax = plt.subplots(figsize=(8, 6))
#====================================================================
ozone_file = '2020-1-8.csv'
#====================================================================
df = pd.read_csv(os.path.join(out_folder, ozone_file))
#====================================================================
world = gpd.read_file(map)

world.plot(ax=ax, color='white', edgecolor='black', linewidth=0.1, alpha=1, legend=True) # GOOD lots the map
#====================================================================
points = [Point(x,y) for x,y in zip(df['LongitudeCenter'], df['LatitudeCenter'])]

#====================================================================
ozone_gdf = gpd.GeoDataFrame(geometry=points).assign(data=df['IntegratedVerticalProfile'])

#====================================================================
# - - - - - - - - - - - Make colorbar for ozone - - - - - - - - - - -
ozone_norm = Normalize(vmin=0, vmax=max(df['IntegratedVerticalProfile']))
ozone_cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'red'], N=200) # Higher N=more smooth

# - - - - - - - - - - - - - Plot ozone data - - - - - - - - - - - - -
ozone_gdf.plot(ax=ax, column='data', cmap=ozone_cmap, norm=ozone_norm, markersize=1, alpha=1, legend=True)
#====================================================================
plt.title(ozone_file)
#====================================================================
plt.show()
#====================================================================

#====================================================================

#====================================================================

#====================================================================