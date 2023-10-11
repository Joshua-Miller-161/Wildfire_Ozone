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
#====================================================================
sys.path.append('/Users/joshuamiller/Documents/Python Files/Wildfire_Ozone')

#====================================================================
from extraction_funcs import ExtractHDF5

#====================================================================
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
    if not file.endswith('hdf5'):
        file_names.remove(file)

#====================================================================
''' Group together files which cover the same days '''
file_dict = {}

for i in range(len(file_names)):
    timestamps = ExtractHDF5(os.path.join(in_folder, file_names[i]), 'Time', groups='all', print_sum=False)
    
    unique_dates_in_file = []
    for timestamp in timestamps['Time']:
        date_obj = parse(timestamp)
        y_m_d = str(date_obj.year) + '-' + str(date_obj.month) + '-' + str(date_obj.day)

        if not (y_m_d in unique_dates_in_file):
            unique_dates_in_file.append(y_m_d)
            #print("Found unique date:", y_m_d, timestamp, unique_dates_in_file)

    for date in unique_dates_in_file:
        try:
            file_dict[date].append(file_names[i])
        except:
            KeyError
            file_dict[date] = [file_names[i]] # First time finding a file that contains this date

print(">> Grouped files")

#====================================================================
''' Create one csv file for each day '''
for date in file_dict.keys():
    df = pd.DataFrame(columns=vars_to_extract)
    df.to_csv(os.path.join(out_folder, date)+'.csv', index=False)
print(">> Initialized csv files")

#====================================================================




#====================================================================
'''
for i in len(file_names):
    
        dict_ = ExtractHDF5(os.path.join(args.in_folder, file_names[i]),
                            vars_to_extract,
                            groups='all',
                            print_sum=False)

        filtered_dict = dict.fromkeys(vars_to_extract)
        for key in filtered_dict.keys():
            filtered_dict[key] = []

        count = 0
        
        for i in range(np.shape(dict_['Time'])[0]): # Shouldn't matter which var you pick
            
            if ((args.min_lat <= dict_['LatitudeCenter'][i]) and (dict_['LatitudeCenter'][i] <= args.max_lat)):
                if ((args.min_lon <= dict_['LongitudeCenter'][i]) and (dict_['LongitudeCenter'][i] <= args.max_lon)):
                    
                    filtered_dict['LongitudeCenter'].append(dict_['LongitudeCenter'][i])
                    filtered_dict['LatitudeCenter'].append(dict_['LatitudeCenter'][i])
                    filtered_dict['Time'].append(dict_['Time'][i])
                    filtered_dict['IntegratedVerticalProfile'].append(dict_['IntegratedVerticalProfile'][i])

                    count += 1

        if count > 0:
            print("Found", count, "points in", file)
            df = pd.DataFrame(filtered_dict)
            df.to_csv(os.path.join(args.out_folder, file[:-5]+'.csv'), sep=",", header=True, index=False)

'''












#====================================================================

fig, ax = plt.subplots(figsize=(8, 6))
#====================================================================
ozone_file = os.listdir(out_folder)[21]
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
plt.title('min. time: '+str(min(df['Time'])) + ', max. time: '+ str(max(df['Time'])))
#====================================================================
plt.show()
#====================================================================

#====================================================================

#====================================================================

#====================================================================