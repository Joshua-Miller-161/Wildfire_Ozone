import sys
sys.dont_write_bytecode = True
import numpy as np
import itertools
import threading
import time
import os
import re
import pandas as pd
import yaml
from sklearn import preprocessing
from scipy.signal import butter, filtfilt
from datetime import datetime, timedelta

sys.path.append(os.getcwd())
from data_utils.preprocessing_funcs import Scale
from ml.ml_utils import ParseModelName
#====================================================================
def FindDate(my_str, start_keyword):

    start_idx = my_str.find(start_keyword)

    my_str = my_str[start_idx+(len(start_keyword)+1):]

    patterns = [r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', # 1776-07-04 00:00:00
                r'\d{4}-\d{2}-\d{2}',                   # 1776-07-04
                r'\d{4}/\d{2}/\d{2}',                   # 1776/07/04
                r'\d{8}']                               # 17760704
    # Search for the pattern in the string
    match = False
    for pattern in patterns:
        match = re.search(pattern, my_str)
    # If the pattern is found, return the matched substring
    if match:
        return match.group()
    # Otherwise, return None
    else:
        return None
#====================================================================
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
#====================================================================
def NumericalDerivative(x, y, method='central', x_to_datetime=True):
    assert np.shape(x) == np.shape(y), "x and y must be same shape. Got x: "+str(np.shape(x))+", y: "+str(np.shape(y))

    start_date = 69
    if isinstance(x[0], datetime): # Convert
        start_date = min(x)
        x = [(date - start_date).days for date in x]
        #print("is datetime", np.shape(x), x)

    if (method == 'central'):
        new_x = np.ones(np.shape(x)[0]-2, float)*-9999
        dy    = np.ones(np.shape(y)[0]-2, float)*-9999
        #print("NumDeriv start: ", start_date, "x:", np.shape(x), ", y:", np.shape(y), ", new_x:", np.shape(new_x), ", dy:", np.shape(dy))
        for i in range(1, np.shape(x)[0]-1):
            new_x[i-1] = x[i]
            dy[i-1]    = (y[i+1]-y[i-1]) / (x[i+1]-x[i-1])
        if x_to_datetime:
            #print("new_x:", np.shape(new_x), new_x)
            new_x = [start_date+timedelta(days=val) for val in new_x]
            #print("new_x:", np.shape(new_x), new_x)
            return new_x, dy
        else:
            return new_x, dy
    
    elif (method == 'forward'):
        new_x = np.ones(np.shape(x)[0]-1, float)*-9999
        dy    = np.ones(np.shape(y)[0]-1, float)*-9999
        for i in range(0, np.shape(x)[0]-1):
            new_x[i] = x[i]
            dy[i]    = (y[i+1]-y[i]) / (x[i+1]-x[i])
        if x_to_datetime:
            new_x = [start_date+timedelta(days=val) for val in new_x]
            return new_x, dy
        else:
            return new_x, dy
    
    elif (method == 'backward'):
        new_x = np.ones(np.shape(x)[0]-1, float)*-9999
        dy    = np.ones(np.shape(y)[0]-1, float)*-9999
        for i in range(1, np.shape(x)[0]):
            new_x[i-1] = x[i]
            dy[i-1]    = (y[i]-y[i-1]) / (x[i]-x[i-1])
        if x_to_datetime:
            new_x = [start_date+timedelta(days=val) for val in new_x]
            return new_x, dy
        else:
            return new_x, dy
    
    else:
        print("method must be 'central', 'forward' or 'backward'.")
#====================================================================
def FFT(t, y, mult_2pi=False):
    # If you believe data is of the form: y = sin(2π * f1 * x) + ...
    #    - set mult_2pi=False to show peak at f1
    # Otherwise if you have: y = sin(f1 * x) + ...
    #    - set mult_2pi=True to show peak at f1
    # Credit: john-hen on Stack Exchange
    n = len(t)
    delta = (max(t) - min(t)) / (n-1)
    k = int(n/2)
    f = np.arange(k) / (n*delta) # 0/(n*delta), 1/(n*delta), 2/(n_delta), ...
    Y = abs(np.fft.rfft(y))[:k]
    if not mult_2pi:
        return (f, Y)
    else:
        return (f*(2*np.pi), Y)
#====================================================================
def DeNoiseFFT(x, y, num_fft, mult_2pi=False, rescale=True):
    (freq, Y) = FFT(x, Scale(y, y), mult_2pi)
    freq_amps = find_largest_values(Y, freq, num_fft)

    y_new = np.zeros_like(y)
    
    for i in range(num_fft):
        freq = freq_amps[i][0]
        amp  = freq_amps[i][1]
        if mult_2pi:
            y_new += amp * np.sin(freq*x)
        else:
            y_new += amp * np.sin(2*np.pi*freq*x)

    if rescale:
        y_new = Scale(y_new, 'minmax', lower=min(y),upper=max(y))
    return y_new
#====================================================================
def ButterLowpassFilter(data, cutoff, sample_rate, order):
    normal_cutoff = cutoff / (.5 * sample_rate)
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
#====================================================================
def Longitude360to180(lon):
    lon = np.asarray(lon)
    return (lon + 180) % 360 - 180
#====================================================================
def Longitude180to360(lon):
    lon = np.asarray(lon)
    return lon % 360   
#====================================================================
def GetBoxCoords(config_path):
    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    min_lat_wa = config['WHOLE_AREA_BOX']['min_lat']
    max_lat_wa = config['WHOLE_AREA_BOX']['max_lat']
    min_lon_wa = config['WHOLE_AREA_BOX']['min_lon']
    max_lon_wa = config['WHOLE_AREA_BOX']['max_lon']

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

    boxes = {'Whole_Area': [min_lon_wa, max_lat_wa, max_lon_wa, min_lat_wa],
             'East_Ocean': [min_lon_eo, max_lat_eo, max_lon_eo, min_lat_eo],
             'West_Ocean': [min_lon_wo, max_lat_wo, max_lon_wo, min_lat_wo],
             'North_Land': [min_lon_nl, max_lat_nl, max_lon_nl, min_lat_nl],
             'South_Land': [min_lon_sl, max_lat_sl, max_lon_sl, min_lat_sl]}
    return boxes
#====================================================================
def CompareDates(a, b):
    date_a = GetDateInStr(a)
    date_b = GetDateInStr(b)
    if date_a == date_b:
        print("MATCH date_a=", date_a, ", date_b=", date_b)
        return True
    else:
        print("NOT MATCH date_a=", date_a, ", date_b=", date_b)
        return False
#====================================================================
def FindIntersection(x_arrays, y_arrays):
    assert len(x_arrays) == len(y_arrays), "Must have same amount of lists in both. Got x_arrays:"+str(len(x_arrays))+', y_arrays:'+str(len(y_arrays))
    for i in range(len(x_arrays)):
        print(np.shape(x_arrays[i]), np.shape(y_arrays[i]))
        assert np.shape(x_arrays[i]) == np.shape(y_arrays[i]), "i="+str(i)+". Shapes must match. Got: x: "+str(np.shape(x_arrays[i]))+", y: "+str(np.shape(y_arrays[i]))

    for i in range(len(x_arrays)):
        x_arrays[i] = np.asarray(x_arrays[i])
    
    x_intersect = x_arrays[0]
    for i in range(1, len(x_arrays)):
        x_intersect = np.intersect1d(x_intersect, x_arrays[i])

    print("x_intersect:", np.shape(x_intersect))

    y_intersects = []
    for i in range(len(y_arrays)):
        x = x_arrays[i]
        y = y_arrays[i]

        if (len(np.shape(y)) > 1):
            y_intersect = np.ones(((np.shape(x_intersect)[0],) + np.shape(y)[1:]), float) * -9999
            for j in range(np.shape(x_intersect)[0]):
                y_intersect[j, ...] = y[np.where(x == x_intersect[j])[0], ...]
            y_intersects.append(y_intersect)
        else:
            #print(" - i:", i, ", y:", y)
            y_intersect = np.ones(np.shape(x_intersect)[0], float) * -9999
            for j in range(np.shape(x_intersect)[0]):
                y_intersect[j] = y[np.where(x == x_intersect[j])[0]]
            y_intersects.append(y_intersect)
            #print(" - i:", i, ", y_int:", y_intersect)

    return x_intersect, y_intersects
#====================================================================
def interpolate_data(dates, values):
    # Convert the list of datetime objects to a pandas Series
    series = pd.Series(values, index=pd.to_datetime(dates))

    # Create a date range for the uninterrupted stream of days
    date_range = pd.date_range(start=min(dates), end=max(dates))

    # Reindex the series to the full date range with linear interpolation for missing values
    series_interp = series.reindex(date_range).interpolate(method='linear')

    # The interpolated time series
    t_interp = series_interp.index
    y_interp = series_interp.values

    return t_interp, y_interp
#====================================================================
def find_largest_values(y, x, num):
    indices = sorted(range(len(y)), key=lambda i: y[i], reverse=True)[:num]
    values = [(x[i], y[i]) for i in indices]
    values = list(sorted(values, key=lambda t: t[0]))
    return values 
#====================================================================
def RecoverScaledTime(config_path, time, start_date=datetime(1970, 1, 1), lower=0, upper=6):
    datetimes = []

    for num in time:
        datetimes.append()
    return datetimes

#====================================================================

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