import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
from matplotlib.colors import Normalize, LogNorm, FuncNorm
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import sys
import numpy as np
import re
from scipy.fft import fft, fftfreq
from datetime import datetime, timedelta
import yaml
    
sys.path.append(os.getcwd())
from data_utils.extraction_funcs import Extract_netCDF4
from misc.misc_utils import GetBoxCoords, PlotBoxes, GetDateInStr, CompareDates, Scale, MinMaxScale, FFT
#====================================================================
base_path    = "/Users/joshuamiller/Documents/Lancaster/Data"
O3_folders   = ['Kriged_L2_O3_TCL', 'East_Ocean/L2__O3_TCL', 'West_Ocean/L2__O3_TCL', 'South_Land/L2__O3_TCL', 'North_Land/L2__O3_TCL']
fire_folders = ['Kriged_MODIS_C61', 'East_Ocean/MODIS_C61', 'West_Ocean/MODIS_C61', 'South_Land/MODIS_C61', 'North_Land/MODIS_C61']
labels       = ['Whole_area', 'East_ocean', 'West_ocean', 'South_land', 'North_land']
region = 0
#====================================================================
''' Get data '''
num_O3_files = 0
for folder in O3_folders:
    for file in os.listdir(os.path.join(base_path, folder)):
        if file.endswith('.csv') or file.endswith('.nc'):
            num_O3_files += 1
    #print("Folder:", folder, num_O3_files / len(O3_folders))
num_O3_files = int(num_O3_files / len(O3_folders))

num_fire_files = 0
for folder in fire_folders:
    for file in os.listdir(os.path.join(base_path, folder)):
        if file.endswith('.csv') or file.endswith('.nc'):
            num_fire_files += 1
    #print("Folder:", folder, num_fire_files / len(fire_folders))
num_fire_files = int(num_fire_files / len(fire_folders))

num_files = min([num_fire_files, num_O3_files])
#--------------------------------------------------------------------
''' Pull data and get means '''
fire_dict = {}
for i in range(len(fire_folders)):
    fire_means = np.ones(num_fire_files, float) * -9999
    fire_dates = ['lolololol'] * num_fire_files
    j = 0
    for fire_file in os.listdir(os.path.join(base_path, fire_folders[i])):
        if fire_file.endswith('.csv'):

            date = GetDateInStr(fire_file)
            #print(fire_file, 'date=',date)
            df = pd.read_csv(os.path.join(os.path.join(base_path, fire_folders[i]), fire_file))
            fire_means[j] = np.mean(df['frp'].values)
            fire_dates[j] = datetime.strptime(date, '%Y%m%d')
            j += 1
    
    fire_dict[labels[i]+'_means'] = fire_means
    fire_dict[labels[i]+'_dates'] = fire_dates
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
O3_dict = {}
for i in range(len(O3_folders)):
    O3_means = np.ones(num_O3_files, float) * -9999
    O3_dates = ['lolololol'] * num_O3_files
    j = 0
    for O3_file in os.listdir(os.path.join(base_path, O3_folders[i])):
        if O3_file.endswith('.nc'):
            dict_ = Extract_netCDF4(os.path.join(os.path.join(base_path, O3_folders[i]), O3_file),
                                   ['ozone_tropospheric_vertical_column', 'dates_for_tropospheric_column'],
                                    groups='all',
                                    print_sum=False)
            ozone = np.squeeze(dict_['ozone_tropospheric_vertical_column'])
            dates_O3 = ''.join(dict_['dates_for_tropospheric_column'])
            dates_O3 = dates_O3.split(' ')
            #print("DATES", dates_O3)
            O3_means[j] = np.mean(ozone.ravel())
            O3_dates[j] = datetime.strptime(dates_O3[0], '%Y%m%d')
            #print(j, O3_dates[j])
            j += 1
        
        elif O3_file.endswith('.csv'):
            date = GetDateInStr(O3_file)

            df = pd.read_csv(os.path.join(os.path.join(base_path, O3_folders[i]), O3_file))
            O3_means[j] = np.mean(df['ozone_tropospheric_vertical_column'].values)
            O3_dates[j] = datetime.strptime(date, '%Y%m%d')
            #print(j, O3_dates[j])
            j += 1

    O3_dict[labels[i]+'_means'] = O3_means
    O3_dict[labels[i]+'_dates'] = O3_dates
    #print(O3_dates)
#--------------------------------------------------------------------
''' Sort all according to date '''
for label in labels:
    O3_dates = O3_dict[label+'_dates']
    O3_means = O3_dict[label+'_means']

    sorted_dt_list = sorted(enumerate(O3_dates), key=lambda x: x[1])
    sorted_idx = [x[0] for x in sorted_dt_list]
    sorted_idx = np.asarray(sorted_idx)
    sorted_dates = [x[1] for x in sorted_dt_list]

    O3_dict[label+'_dates'] = sorted_dates
    O3_dict[label+'_means'] = O3_means[sorted_idx]
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    fire_dates = fire_dict[label+'_dates']
    fire_means = fire_dict[label+'_means']

    sorted_dt_list = sorted(enumerate(fire_dates), key=lambda x: x[1])
    sorted_idx = [x[0] for x in sorted_dt_list]
    sorted_idx = np.asarray(sorted_idx)
    sorted_dates = [x[1] for x in sorted_dt_list]

    fire_dict[label+'_dates'] = sorted_dates
    fire_dict[label+'_means'] = fire_means[sorted_idx]
#====================================================================
fig, ax = plt.subplots(5,1,figsize=(12, 7), sharex=True)
fig.subplots_adjust(hspace=0, wspace=0)

def Plot_O3_Fire(ax, fire_x, fire, O3_x, O3, 
                 fire_color='red', O3_color='blue', alpha=1):
    ax.set_xlabel('Date')
    ax.set_ylabel('Avg. FRP (MW)', color=fire_color)
    ax.plot(fire_x, fire, color=fire_color, linestyle='-', linewidth=1, alpha=alpha)
    ax.tick_params(axis='y', labelcolor=fire_color)

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel(r'$O_3$'+' '+r'$\left(\frac{mol}{m^2}\right)$', color=O3_color)  # we already handled the x-label with ax
    ax2.plot(O3_x, O3, color=O3_color, linestyle='-', linewidth=1, alpha=alpha)
    ax2.tick_params(axis='y', labelcolor=O3_color)

def FindIntersection(x1, y1, x2, y2):
    assert np.shape(x1) == np.shape(y1), "Shapes must match. Got: x1: "+str(np.shape(x1))+", y1: "+str(np.shape(y1))
    assert np.shape(x2) == np.shape(y2), "Shapes must match. Got: x2: "+str(np.shape(x2))+", y2: "+str(np.shape(y2))

    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    #print(type(x1), np.shape(x1),type(x2), np.shape(x2))
    x_intersect = np.intersect1d(x1, x2)
    print(np.shape(x_intersect))

    y1_intersect = 0
    y2_intersect = 0
    if (len(np.shape(y1)) > 1):
        y1_intersect = np.ones(((np.shape(x_intersect)[0],) + np.shape(y1)[1:]), float) * -9999
        for i in range(np.shape(x_intersect)[0]):
            y1_intersect[i, ...] = y1[np.where(x1 == x_intersect[i])[0], ...]
    else:
        y1_intersect = np.ones(np.shape(x_intersect)[0], float) * -9999
        for i in range(np.shape(x_intersect)[0]):
            y1_intersect[i] = y1[np.where(x1 == x_intersect[i])[0]]

    if (len(np.shape(y2)) > 1):
        y2_intersect = np.ones(((np.shape(x_intersect)[0],) + np.shape(y2)[1:]), float) * -9999
        for i in range(np.shape(x_intersect)[0]):
            y2_intersect[i, ...] = y2[np.where(x2 == x_intersect[i])[0], ...]
    else:
        y2_intersect = np.ones(np.shape(x_intersect)[0], float) * -9999
        for i in range(np.shape(x_intersect)[0]):
            idx = np.where(x2 == x_intersect[i])[0]
            if (np.shape(idx)[0] > 1):
                for lol in idx:
                    print("AHHHHHH", x2[lol])
            y2_intersect[i] = y2[np.where(x2 == x_intersect[i])[0]]

    return x_intersect, y1_intersect, y2_intersect

def PlotWindow(ax, x1, x2, ymin, ymax, color, alpha, label):
    ax.axvspan(x1, x2, ymin, ymax, color=color, alpha=alpha, label=label)
    ax.axvline(x1, ymin, ymax, color=color, alpha=1, label=label)
    ax.axvline(x1, ymin, ymax, color=color, alpha=1, label=label)

for i in range(5):
    new_dates, new_fire, new_O3 = FindIntersection(fire_dict[labels[i]+'_dates'],
                                                   fire_dict[labels[i]+'_means'],
                                                   O3_dict[labels[i]+'_dates'],
                                                   O3_dict[labels[i]+'_means'])
    Plot_O3_Fire(ax[i], new_dates, new_fire, new_dates, new_O3)
    
    ax[i].set_xlim(datetime(2018, 1, 1, 0, 0, 0), datetime(2023, 1, 1, 0, 0, 0))

    corr_string = r'$\rho=$'+str(round(np.corrcoef(new_fire, new_O3)[0][1], 3))

    if ('ocean' in labels[i]):
        ax[i].set_ylim(-0.1, 10)
        ax[i].text(datetime(2018, 2, 1, 0, 0, 0), 7, labels[i]+'\n'+corr_string)
    else:
        ax[i].text(datetime(2018, 2, 1, 0, 0, 0), 
                  .7 * max(fire_dict[labels[i]+'_means']), 
                  labels[i]+'\n'+corr_string)

    #PlotWindow(ax[i], datetime(2018, 1, 1, 0, 0, 0), datetime(2018, 4, 1, 0, 0, 0), 0, 100, 'lightcoral', .5, 'fire')
    #PlotWindow(ax[i], datetime(2018, 3, 1, 0, 0, 0), datetime(2018, 6, 1, 0, 0, 0), 0, 100, 'cyan', .5, 'O3')
#====================================================================
windowsizes = [15, 30, 60, 120]
overlaps    = [0, .5]
colors      = ['springgreen', 'violet', 'royalblue']


fig2, ax2 = plt.subplots(1+len(windowsizes),1,figsize=(12, 7), sharex=True)
fig2.subplots_adjust(hspace=0, wspace=0)

new_dates, new_fire, new_O3 = FindIntersection(fire_dict[labels[region]+'_dates'],
                                               fire_dict[labels[region]+'_means'],
                                               O3_dict[labels[region]+'_dates'],
                                               O3_dict[labels[region]+'_means'])

Plot_O3_Fire(ax2[0], new_dates, new_fire, new_dates, new_O3)

for i in range(len(windowsizes)):
    ax2[i+1].plot(new_dates, np.zeros_like(new_dates), 'k--', alpha=.5)
    for j in range(len(overlaps)):

        offset = int(windowsizes[i] * (1 - overlaps[j]))
        num_corrs = np.shape(new_dates)[0] - offset - windowsizes[i]
        corrs = np.ones(num_corrs, float) * -9999

        ax2[i+1].axvspan(new_dates[j * 365], new_dates[j * 365]+timedelta(days=windowsizes[i]), 0, 1, color=colors[j], alpha=.1)
        ax2[i+1].axvspan(new_dates[j * 365]+timedelta(days=offset), new_dates[j * 365]+timedelta(days=windowsizes[i])+timedelta(days=offset), 0, 1, color=colors[j], alpha=.1)

        for k in range(num_corrs):
            corrs[k] = np.corrcoef(new_fire[k:(k+windowsizes[i])], new_O3[(k+offset):(k+windowsizes[i]+offset)])[0][1]
            #print('w=', windowsizes[i], ', o=', offset)
            #print(k, new_dates[k], new_dates[k+windowsizes[i]],'|', new_dates[k+offset], new_dates[k+windowsizes[i]+offset])

        ax2[i+1].plot(new_dates[:num_corrs], corrs, color=colors[j], label=str(100*overlaps[j])+'% overlap')        
        ax2[i+1].set_ylim(-1.25, 1.25)
        ax2[i+1].set_ylabel('Correlation')
        ax2[i+1].legend(loc='upper right')
        ax2[i+1].text(datetime(2018, 2, 1, 0, 0, 0), .95, 'Window size: '+str(windowsizes[i])+' days')

    ax2[i+1].set_xlim(datetime(2018, 1, 1, 0, 0, 0), datetime(2023, 1, 1, 0, 0, 0))

ax2[0].set_xlim(datetime(2018, 1, 1, 0, 0, 0), datetime(2023, 1, 1, 0, 0, 0))
ax2[0].set_title(labels[region])

#====================================================================
intersect_dict = {}
for label in labels:
    if not ((label=='East_ocean') or (label=='West_ocean')):
        new_dates, new_fire, new_O3 = FindIntersection(fire_dict[label+'_dates'],
                                                       fire_dict[label+'_means'],
                                                       O3_dict[label+'_dates'],
                                                       O3_dict[label+'_means'])
        intersect_dict[label+'_fire_means']  = new_fire
        intersect_dict[label+'_O3_means']    = new_O3
        intersect_dict[label+'_dates'] = new_dates
#====================================================================
fig3, ax3 = plt.subplots(3, 1, figsize=(12, 7))
fig3.subplots_adjust(hspace=1, wspace=0)

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

def find_largest_values(y, x, num):
    indices = sorted(range(len(y)), key=lambda i: y[i], reverse=True)[:num]
    values = [(x[i], y[i]) for i in indices]
    values = list(sorted(values, key=lambda t: t[0]))
    return values
#--------------------------------------------------------------------
start_date = min(intersect_dict[labels[region]+'_dates'])
T = np.asarray([(date - start_date).days for date in intersect_dict[labels[region]+'_dates']])
O3 = np.asarray(intersect_dict[labels[region]+'_O3_means'])
fire = np.asarray(intersect_dict[labels[region]+'_fire_means'])

T_interp, O3_interp = interpolate_data(intersect_dict[labels[region]+'_dates'], O3)
T_interp, fire_interp = interpolate_data(intersect_dict[labels[region]+'_dates'], fire)
start_date = min(T_interp)
T = np.asarray([(date - start_date).days for date in T_interp])
print(np.shape(T_interp), np.shape(O3_interp), np.shape(fire_interp))
#--------------------------------------------------------------------
(freq_O3, Y_O3) = FFT(T, Scale(O3_interp, O3_interp), True)
Y_scaled_O3 = MinMaxScale(Y_O3, 0, 1)
#print(PSD)
ax3[1].scatter(freq_O3, Y_scaled_O3, s=16, facecolors='blue')
ax3[1].set_xlabel('Frequency')
#ax3[1].set_yscale('log')
ax3[1].set_xlim(0, max(freq_O3))
ax3[1].grid(True)

vals = find_largest_values(Y_scaled_O3, freq_O3, 5)
text = ''
for lol in vals:
    text += '(' + str(round(lol[0], 3)) +', ' + str(round(lol[1], 3)) + ') '
    ax3[1].scatter(lol[0], lol[1], s=48, facecolors='blue', edgecolors='black')
ax3[1].set_title(text)
#--------------------------------------------------------------------
(freq_fire, Y_fire) = FFT(T, Scale(fire_interp, fire_interp), True)
Y_scaled_fire = MinMaxScale(Y_fire, 0, 1)

ax3[2].scatter(freq_fire, Y_scaled_fire, s=16, facecolors='red')
ax3[2].set_xlabel('Frequency')
#ax3[2].set_yscale('log')
ax3[2].set_xlim(0, max(freq_fire))
ax3[2].grid(True)

vals = find_largest_values(Y_scaled_fire, freq_fire, 5)
text = ''
for lol in vals:
    text += '(' + str(round(lol[0], 3)) +', ' + str(round(lol[1], 3)) + ') '
    ax3[2].scatter(lol[0], lol[1], s=48, facecolors='red', edgecolors='black')
ax3[2].set_title(text)
#--------------------------------------------------------------------
Plot_O3_Fire(ax3[0],
             intersect_dict[labels[region]+'_dates'], fire, 
             intersect_dict[labels[region]+'_dates'], O3)
ax3[0].set_title(labels[region])
#--------------------------------------------------------------------


# from statsmodels.tsa.seasonal import seasonal_decompose
# series = O3_interp
# result = seasonal_decompose(series, model='additive', period=365)
# result.plot()


# series = fire_interp
# result = seasonal_decompose(series, model='additive', period=365)
# result.plot()

plt.show()