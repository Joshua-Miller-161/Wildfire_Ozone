import numpy as np
import itertools
import threading
import time
import sys
import re
import pandas as pd

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