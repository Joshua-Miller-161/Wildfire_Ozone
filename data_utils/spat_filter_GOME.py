import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import sys
import os
import csv
#====================================================================
sys.path.append('/Users/joshuamiller/Documents/Python Files/Wildfire_Ozone')
#====================================================================
from extraction_funcs import ExtractHDF5
#====================================================================
parser = argparse.ArgumentParser(description='Get file locations')
parser.add_argument('--in_folder', nargs=1, type=str, help='location of the GOME data files',
                    default="/Users/joshuamiller/Documents/Lancaster/Data/Gome")
parser.add_argument('--out_folder', nargs=1, type=str, help='location to put the filtered files',
                    default="/Users/joshuamiller/Documents/Lancaster/Data/Filtered_GOME")
parser.add_argument('--map', nargs=1, type=str, help='location of the world map shape file',
                    default="/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")
args = parser.parse_args()
#====================================================================
for file in os.listdir(args.in_folder):
    if file.endswith('hdf5'):
        dict_ = dict_ = ExtractHDF5(args.data,
                                    ['LatitudeCenter', 'LongitudeCenter', 'Time', 'IntegratedVerticalProfile'],
                                    groups=['DATA', 'GEOLOCATION'],
                                    print_sum=True)
#====================================================================

#====================================================================

#====================================================================

#====================================================================

#====================================================================

#====================================================================

#====================================================================

#====================================================================

#====================================================================
my_dict = {"name": "Alice", "age": 25, "gender": "F"}

# Open a CSV file in write mode
with open("my_dict.csv", "w") as f:
    # Create a DictWriter object with the keys of the dictionary as field names
    writer = csv.DictWriter(f, fieldnames=my_dict.keys())
    # Write the header row with the field names
    writer.writeheader()
    # Write the dictionary as a row
    writer.writerow(my_dict)
#====================================================================