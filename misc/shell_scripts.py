import os
import re
import zipfile
import sys
import numpy as np
import shutil
from pathlib import Path

sys.path.append(os.getcwd())
from data_utils.extraction_funcs import Extract_netCDF4


def CombO3Dir(base_path, pattern=r"\(\d+\)", delete=False):
    # Create an empty list to store the matched files
    matched_files = []

    # Loop through the files in the directory
    for filename in os.listdir(base_path):
        # Check if the filename matches the pattern
        if re.search(pattern, filename):
            # Add the file to the matched list
            matched_files.append(filename)

    # Print the list of matched files
    print("The following files match the pattern:")
    for file in matched_files:
        print(file)

    if delete:
        # Delete the matched files
        for file in matched_files:
            # Get the full path of the file
            file_path = os.path.join(base_path, file)
            # Delete the file
            os.remove(file_path)

        # Print a confirmation message
        print("The matched files have been deleted.")


def UnZipAndDel(base_path):
    # Define a function that unzips a zip file
    def unzip_file(zip_file):
        # Get the full path of the zip file
        zip_path = os.path.join(base_path, zip_file)
        # Create a ZipFile object
        with zipfile.ZipFile(zip_path, "r") as zip_obj:
            # Extract all the contents of the zip file to the same base_path
            zip_obj.extractall(base_path)
            # Print a confirmation message
            print(f"{zip_file} has been unzipped.")

    zip_files = []

    for filename in os.listdir(base_path):
        print(filename[-4:])
        if (filename[-4:]=='.zip'):

            zip_files.append(os.path.join(base_path, filename))

            unzip_file(os.path.join(base_path, filename))
        
    return zip_files


def DelZip(base_path):
    for filename in os.listdir(base_path):
        if (filename[-4:] == '.zip'):
            os.remove(os.path.join(base_path, filename))


def Filter_OFFL_RPRO(base_path, radius, delete=False):
    OFFL_start_dates = []
    OFFL_file_names  = []

    RPRO_start_dates = []
    RPRO_file_names  = []
    #================================================================
    folders = os.listdir(base_path)

    for folder in folders:
        if not os.path.isdir(os.path.join(base_path, folder)):
            folders.remove(folder)
    #================================================================
    for folder in folders:
        files = os.listdir(os.path.join(base_path, folder))
        for file in files:
            if file.endswith('.nc'):
                nc_file = os.path.join(os.path.join(base_path, folder), file)
                date_dict = Extract_netCDF4(nc_file, 
                                            var_names=['dates_for_tropospheric_column', 'time'], 
                                            groups='all', 
                                            print_sum=False)
                
                date = date_dict['dates_for_tropospheric_column'].split(' ')[0]
                #print(date_dict['dates_for_tropospheric_column'].split(' ')[0], type(date_dict['dates_for_tropospheric_column']))
                #print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
                #====================================================
                if 'OFFL' in file:
                    OFFL_start_dates.append(date)
                    OFFL_file_names.append(nc_file)
                
                elif 'RPRO' in file:
                    RPRO_start_dates.append(date)
                    RPRO_file_names.append(nc_file)

                else:
                    print("AHHHHHHH", nc_file)
    #================================================================
    print("len(RPRO_file_names)=", len(RPRO_file_names), ", len(OFFL_file_names)=", len(OFFL_file_names))

    #================================================================
    for date in RPRO_start_dates:
        for i in range(len(OFFL_start_dates)):
            if (date == OFFL_start_dates[i]):
                print("AHHHHHHH", date, i)
                if delete:
                    print("Removing:", OFFL_file_names[i])
                    shutil.rmtree(Path(OFFL_file_names[i]).parent)
    #================================================================


def DelEmptyFolders(base_path):
    folders = os.listdir(base_path)

    for folder in folders:
        if not os.path.isdir(os.path.join(base_path, folder)):
            folders.remove(folder)
    #================================================================
    for folder in folders:
        files = os.listdir(os.path.join(base_path, folder))
        has_nc = False
        for file in files:
            if file.endswith('.nc'):
                has_nc = True
        
        if not has_nc:
            shutil.rmtree(os.path.join(base_path, folder))


def DelDupDays(base_path):
    folders = os.listdir(base_path)
    for folder in folders:
        if not os.path.isdir(os.path.join(base_path, folder)):
            folders.remove(folder)

    dates = []
    paths = []

    for folder in folders:
        files = os.listdir(os.path.join(base_path, folder))
        for file in files:
            if file.endswith('.nc'):
                nc_file = os.path.join(os.path.join(base_path, folder), file)
                date_dict = Extract_netCDF4(nc_file, 
                                            var_names=['dates_for_tropospheric_column', 'time'], 
                                            groups='all', 
                                            print_sum=False)
                
                date = date_dict['dates_for_tropospheric_column'].split(' ')[0]
                dates.append(date)
                paths.append(nc_file)
    
    dates = np.asarray(dates)
    #print(dates)
    #================================================================
    idx_to_delete = []
    i = 0
    while(i < len(dates)):
        idx = np.where(dates == dates[i])[0]
        print(dates[i], i, idx, np.shape(idx))
        if (np.shape(idx)[0] > 1):
            print("---", date, i, idx)
            for j in range(1, np.shape(idx)[0]):
                if not (idx[j] in idx_to_delete):
                    idx_to_delete.append(idx[j])
                    i += 1
        i += 1

    print(idx_to_delete)
    #================================================================
    for idx in idx_to_delete:
        file = paths[idx]
        shutil.rmtree(Path(file).parent)




        
    
#====================================================================
CombO3Dir("/Users/joshuamiller/Documents/Lancaster/Data/L2_O3_TCL", delete=True)

zip_files = UnZipAndDel("/Users/joshuamiller/Documents/Lancaster/Data/L2_O3_TCL")

DelZip("/Users/joshuamiller/Documents/Lancaster/Data/L2_O3_TCL")

Filter_OFFL_RPRO("/Users/joshuamiller/Documents/Lancaster/Data/L2_O3_TCL", 10, delete=False)

DelEmptyFolders("/Users/joshuamiller/Documents/Lancaster/Data/L2_O3_TCL")

DelDupDays("/Users/joshuamiller/Documents/Lancaster/Data/L2_O3_TCL")