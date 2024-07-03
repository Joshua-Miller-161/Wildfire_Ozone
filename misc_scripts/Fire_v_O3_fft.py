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
from datetime import datetime, timedelta
from scipy.fft import fft, ifft
    
sys.path.append(os.getcwd())
from data_utils.extraction_funcs import Extract_netCDF4
from misc.misc_utils import GetDateInStr, Scale, MinMaxScale, FFT, DeNoiseFFT, Plot_O3_Fire, FindIntersection, interpolate_data, find_largest_values
from vis.plotting_utils import ShowYearMonth
#====================================================================
base_path    = "/Users/joshuamiller/Documents/Lancaster/Data"
O3_folders   = ['Kriged_L2_O3_TCL', 'East_Ocean/L2__O3_TCL', 'West_Ocean/L2__O3_TCL', 'South_Land/L2__O3_TCL', 'North_Land/L2__O3_TCL']
fire_folders = ['Kriged_MODIS_C61', 'East_Ocean/MODIS_C61', 'West_Ocean/MODIS_C61', 'South_Land/MODIS_C61', 'North_Land/MODIS_C61']
labels       = ['Whole_area', 'East_ocean', 'West_ocean', 'South_land', 'North_land']
region = 0
num_fft = 50
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
intersect_dict = {}
for label in labels:
    new_dates, new_fire, new_O3 = FindIntersection(fire_dict[label+'_dates'],
                                                    fire_dict[label+'_means'],
                                                    O3_dict[label+'_dates'],
                                                    O3_dict[label+'_means'])
    intersect_dict[label+'_fire_means']  = new_fire
    intersect_dict[label+'_O3_means']    = new_O3
    intersect_dict[label+'_dates'] = new_dates
#====================================================================
fig, ax = plt.subplots(len(labels),1,figsize=(12, 7), sharex=True)
fig.subplots_adjust(hspace=0, wspace=0)

for i in range(len(labels)):
    dates = intersect_dict[labels[i]+'_dates']
    O3    = intersect_dict[labels[i]+'_O3_means']
    fire  = intersect_dict[labels[i]+'_fire_means']

    Plot_O3_Fire(ax[i], dates, fire, dates, O3)
    
    ax[i].set_xlim(datetime(2018, 1, 1, 0, 0, 0), datetime(2023, 1, 1, 0, 0, 0))

    ShowYearMonth(ax[i], dates)
    
    corr_string = r'$\rho=$'+str(round(np.corrcoef(fire, O3)[0][1], 3))

    if ('ocean' in labels[i]):
        ax[i].set_ylim(-0.1, 10)
        ax[i].text(datetime(2018, 2, 1, 0, 0, 0), 7, labels[i]+'\n'+corr_string)
    else:
        ax[i].text(datetime(2018, 2, 1, 0, 0, 0), 
                  .7 * max(fire_dict[labels[i]+'_means']), 
                  labels[i]+'\n'+corr_string)     
#====================================================================
        


#====================================================================
        

#====================================================================
        
#====================================================================
        
#====================================================================
fig3, ax3 = plt.subplots(4, 1, figsize=(12, 7))
fig3.subplots_adjust(hspace=1, wspace=0)

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
ax3[2].scatter(freq_O3, Y_scaled_O3, s=16, facecolors='blue')
ax3[2].set_xlabel('Frequency')
#ax3[2].set_yscale('log')
ax3[2].set_xlim(0, max(freq_O3))
ax3[2].grid(True)

vals_O3 = find_largest_values(Y_scaled_O3, freq_O3, num_fft)
text = ''
for lol in vals_O3:
    #text += '(' + str(round(lol[0], 3)) +', ' + str(round(lol[1], 3)) + ') '
    ax3[3].vlines(lol[0], 0, 1, color='blue', lw=1)
    ax3[2].scatter(lol[0], lol[1], s=48, facecolors='blue', edgecolors='black')
    
ax3[2].set_title(text)
#--------------------------------------------------------------------
(freq_fire, Y_fire) = FFT(T, Scale(fire_interp, fire_interp), True)
Y_scaled_fire = MinMaxScale(Y_fire, 0, 1)

ax3[3].scatter(freq_fire, Y_scaled_fire, s=16, facecolors='red')
ax3[3].set_xlabel('Frequency')
#ax3[3].set_yscale('log')
ax3[3].set_xlim(0, max(freq_fire))
ax3[3].grid(True)

vals_fire = find_largest_values(Y_scaled_fire, freq_fire, num_fft)
text = ''
for lol in vals_fire:
    #text += '(' + str(round(lol[0], 3)) +', ' + str(round(lol[1], 3)) + ') '
    ax3[2].vlines(lol[0], 0, 1, color='red', lw=1)
    ax3[3].scatter(lol[0], lol[1], s=48, facecolors='red', edgecolors='black')
ax3[3].set_title(text)
#--------------------------------------------------------------------
Plot_O3_Fire(ax3[0],
             intersect_dict[labels[region]+'_dates'], fire, 
             intersect_dict[labels[region]+'_dates'], O3)
ax3[0].set_title(labels[region])
ax3[0].set_xlim(datetime(2018, 1, 1, 0, 0, 0), datetime(2023, 1, 1, 0, 0, 0))
#--------------------------------------------------------------------
new_fire = DeNoiseFFT(T, fire_interp, num_fft, True)
new_O3   = DeNoiseFFT(T, O3_interp, num_fft, True)

Plot_O3_Fire(ax3[1], T_interp, new_fire, T_interp, new_O3)

ax3[1].set_xlim(datetime(2018, 1, 1, 0, 0, 0), datetime(2023, 1, 1, 0, 0, 0))

corr_string = r'$\rho=$'+str(round(np.corrcoef(new_fire, new_O3)[0][1], 3))
# ax3[1].text(datetime(2018, 2, 1, 0, 0, 0), 
#                     .7 * max(fire_dict[labels[region]+'_means']), 
#                     labels[region]+'\n'+corr_string)
plt.show()