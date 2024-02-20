import numpy as np
import itertools
import threading
import time
import sys
import re
import pandas as pd
import yaml
from sklearn import preprocessing

def Funnel(start_size, end_size, r=np.e):
    assert not start_size == end_size, "'start_size' and 'end_size' must be different. Got start_size = {}, output_size = {}".format(start_size, end_size)
    if (start_size > end_size):
        if (round(start_size / r) <= end_size):
            return [start_size, end_size]

        else:
            sizes = [start_size]
            i = 1
            while ((round(start_size / r**i)) > end_size):
                sizes.append(round(start_size / r**i))
                i += 1
            sizes.append(end_size)
            return sizes
    
    elif (start_size < end_size):
        if (start_size * r > end_size):
            return [start_size, end_size]
        
        else:
            sizes = [start_size]
            i = 1
            while ((round(start_size * r**i)) < end_size):
                sizes.append(round(start_size * r**i))
                i += 1
            sizes.append(end_size)
            return sizes
        

def DownSample(data, downsample_rate, axis, delete=False):
    '''
    Made by ChatGPT - Extracts data points separated by skip along the given axis

    Returns downsampled data.
    
    - data (ndarray) - the data
    - skip (int) - the number of elements that are skiped when downsampling
    - axis (int) - the axis on which to downsample
    - delete (bool, optional) - whether or not to delete the original data in order to save memory
    '''
    slices       = [slice(None)] * data.ndim
    slices[axis] = slice(None, None, downsample_rate)
    new_data     = data[tuple(slices)]
    
    print('Orig. shape :', np.shape(data), "----> new shape :", np.shape(new_data))

    if delete:
        del(data)

    return new_data

def Scale(data, reference, method='standard'):
    '''
    This function scales the data, either using the StandardScaler or MaxAbsScaler
    :param data: nparray, the data to be scaled 
    :param reference: nparray, what to use as a reference for the scaler, must be same size as data
    :param method: str, either 'standard' or 'maxabs', chooses the scaling method
    '''
    data = np.array(data)
    reference = np.array(reference)

    orig_shape = np.shape(data)

    if method == 'standard':
        scaler = preprocessing.StandardScaler() # Makes mean = 0, stdev = 1
    elif method == 'maxabs':
        scaler = preprocessing.MaxAbsScaler() # Scales to range [-1, 1], best for sparse data
    else:
        raise ValueError('Invalid method specified. Allowed values are "standard" and "maxabs".')

    scaler.fit(reference.ravel().reshape(-1, 1)) # Flatten array in order to obtain the mean over all the space and time

    scaled_data = scaler.transform(data.ravel().reshape(-1, 1))

    return scaled_data.reshape(orig_shape)

def FindDate(my_str, start_keyword):

    start_idx = my_str.find(start_keyword)

    my_str = my_str[start_idx+(len(start_keyword)+1):]
    # Search for the pattern in the string
    match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', my_str)
    # If the pattern is found, return the matched substring
    if match:
        return match.group()
    # Otherwise, return None
    else:
        return None
    
def Longitude360to180(lon):
    lon = np.asarray(lon)
    return (lon + 180) % 360 - 180

def Longitude180to360(lon):
    lon = np.asarray(lon)
    return lon % 360

def PlotBoxes(config_path, ax, plot_text=False):
    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    min_lat_1 = config['EAST_OCEAN_BOX']['min_lat']
    max_lat_1 = config['EAST_OCEAN_BOX']['max_lat']
    min_lon_1 = config['EAST_OCEAN_BOX']['min_lon']
    max_lon_1 = config['EAST_OCEAN_BOX']['max_lon']

    min_lat_2 = config['WEST_OCEAN_BOX']['min_lat']
    max_lat_2 = config['WEST_OCEAN_BOX']['max_lat']
    min_lon_2 = config['WEST_OCEAN_BOX']['min_lon']
    max_lon_2 = config['WEST_OCEAN_BOX']['max_lon']

    min_lat_3 = config['NORTH_LAND_BOX']['min_lat']
    max_lat_3 = config['NORTH_LAND_BOX']['max_lat']
    min_lon_3 = config['NORTH_LAND_BOX']['min_lon']
    max_lon_3 = config['NORTH_LAND_BOX']['max_lon']

    min_lat_4 = config['SOUTH_LAND_BOX']['min_lat']
    max_lat_4 = config['SOUTH_LAND_BOX']['max_lat']
    min_lon_4 = config['SOUTH_LAND_BOX']['min_lon']
    max_lon_4 = config['SOUTH_LAND_BOX']['max_lon']

    box1 = np.array([[min_lon_1, max_lon_1, max_lon_1, min_lon_1, min_lon_1], 
                    [max_lat_1, max_lat_1, min_lat_1, min_lat_1, max_lat_1]])
    box2 = np.array([[min_lon_2, max_lon_2, max_lon_2, min_lon_2, min_lon_2], 
                    [max_lat_2, max_lat_2, min_lat_2, min_lat_2, max_lat_2]])
    box3 = np.array([[min_lon_3, max_lon_3, max_lon_3, min_lon_3, min_lon_3], 
                    [max_lat_3, max_lat_3, min_lat_3, min_lat_3, max_lat_3]])
    box4 = np.array([[min_lon_4, max_lon_4, max_lon_4, min_lon_4, min_lon_4], 
                    [max_lat_4, max_lat_4, min_lat_4, min_lat_4, max_lat_4]])

    ax.plot(box1[0, :], box1[1, :], 'k-')
    ax.plot(box2[0, :], box2[1, :], 'k-')
    ax.plot(box3[0, :], box3[1, :], 'k-')
    ax.plot(box4[0, :], box4[1, :], 'k-')

    if plot_text:
        ax.text(box1[0, 0], 1+box1[1, 0], 'west ocean')
        ax.text(box2[0, 0], 1+box2[1, 0], 'east ocean')
        ax.text(box3[0, 0], 1+box3[1, 0], 'north land')
        ax.text(box4[0, 0], 1+box4[1, 0], 'south land')

def GetBoxCoords(config_path):
    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    min_lat_eo = config['EAST_OCEAN_BOX']['min_lat']
    max_lat_eo = config['EAST_OCEAN_BOX']['max_lat']
    min_lon_eo = config['EAST_OCEAN_BOX']['min_lon']
    max_lon_eo = config['EAST_OCEAN_BOX']['max_lon']

    min_lat_wo = config['WEST_OCEAN_BOX']['min_lat']
    max_lat_wo = config['WEST_OCEAN_BOX']['max_lat']
    min_lon_wo = config['WEST_OCEAN_BOX']['min_lon']
    max_lon_wo = config['WEST_OCEAN_BOX']['max_lon']

    min_lat_nl = config['NORTH_LAND_BOX']['min_lat']
    max_lat_nl = config['NORTH_LAND_BOX']['max_lat']
    min_lon_nl = config['NORTH_LAND_BOX']['min_lon']
    max_lon_nl = config['NORTH_LAND_BOX']['max_lon']

    min_lat_sl = config['SOUTH_LAND_BOX']['min_lat']
    max_lat_sl = config['SOUTH_LAND_BOX']['max_lat']
    min_lon_sl = config['SOUTH_LAND_BOX']['min_lon']
    max_lon_sl = config['SOUTH_LAND_BOX']['max_lon']

    boxes = {'East_Ocean': [min_lon_eo, max_lat_eo, max_lon_eo, min_lat_eo],
             'West_Ocean': [min_lon_wo, max_lat_wo, max_lon_wo, min_lat_wo],
             'North_Land': [min_lon_nl, max_lat_nl, max_lon_nl, min_lat_nl],
             'South_Land': [min_lon_sl, max_lat_sl, max_lon_sl, min_lat_sl]}
    return boxes

def GetDateInStr(string):
    # Define a regular expression pattern to match dates
    patterns = [r"(\d{4})/(\d{2})/(\d{2})",
                r"(\d{4})-(\d{2})-(\d{2})",
                r"(\d{8})"]
    # Use re.search to find the first match in each string
    date = ''
    for pattern in patterns:
        match = re.search(pattern, string)
        if match:
            date = match.group()
    date = re.sub(r"\D", "", date)
    return date

def CompareDates(a, b):
    date_a = GetDateInStr(a)
    date_b = GetDateInStr(b)
    if date_a == date_b:
        print("MATCH date_a=", date_a, ", date_b=", date_b)
        return True
    else:
        print("NOT MATCH date_a=", date_a, ", date_b=", date_b)
        return False

'''
def ProgressWheel():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write("\rloading " + c)
        sys.stdout.flush()
        time.sleep(0.05)
    sys.stdout.write("\r            \r")
    sys.stdout.flush()
    sys.stdout.write("\rDone!")

t = threading.Thread(target=ProgressWheel)
t.start()
'''



# df = pd.DataFrame({'date': ['06/01/2020', '06/01/2020', '06/03/2020'], 'data': [25, 26, 27]})

# def CreateDateDfs(df):
#     # Create an empty dictionary to store the new dataframes
#     date_dfs = {}
#     # Loop through the unique values of the date column
#     for date in df['date'].unique():
#         # Filter the dataframe by the date value
#         date_df = df[df['date'] == date]
#         # Add the filtered dataframe to the dictionary with the date as the key
#         date_dfs[date] = date_df
#     # Return the dictionary of new dataframes
#     return date_dfs

# lol = CreateDateDfs(df)
# print(lol)