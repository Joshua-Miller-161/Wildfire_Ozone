import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import sys
import numpy as np
from datetime import datetime, timedelta
    
sys.path.append(os.getcwd())
from data_utils.extraction_funcs import Extract_netCDF4
from misc.misc_utils import GetDateInStr, FindIntersection, ButterLowpassFilter
from vis.plotting_utils import ShowYearMonth, Plot_O3_Fire
#====================================================================
base_path    = "/Users/joshuamiller/Documents/Lancaster/Data"
O3_folders   = ['Kriged_L2_O3_TCL', 'East_Ocean/L2__O3_TCL', 'West_Ocean/L2__O3_TCL', 'South_Land/L2__O3_TCL', 'North_Land/L2__O3_TCL']
fire_folders = ['Kriged_MODIS_C61', 'East_Ocean/MODIS_C61', 'West_Ocean/MODIS_C61', 'South_Land/MODIS_C61', 'North_Land/MODIS_C61']
labels       = ['Whole_area', 'East_ocean', 'West_ocean', 'South_land', 'North_land']
region = 0
cutoff = 0.016
order = 4
subtract_region = 1
new_regions = ['West_ocean', 'East_ocean']
fire_region = 'South_land'
lag_days = [0, 5, 10, 30]
colors = ['blue', 'darkorange', 'green', 'blueviolet']
#====================================================================
def Format(ax, dates, O3_color, fire_label, O3_label):
    ax.set_xlim(datetime(2017, 1, 1, 0, 0, 0), datetime(2023, 1, 1, 0, 0, 0))
    ShowYearMonth(ax, dates)
    legend_fire = mlines.Line2D([], [], color='red', linestyle='-', label=fire_label)
    legend_O3   = mlines.Line2D([], [], color=O3_color, linestyle='-', label=O3_label)
    ax.legend(handles = [legend_fire, legend_O3], loc='upper left')

def NormalPlot(ax, dates, O3_color, fire_label, O3_label):
    dates = intersect_dict[new_regions[i]+'_dates']
    O3    = intersect_dict[new_regions[i]+'_O3_means']
    Plot_O3_Fire(ax, dates, intersect_dict[fire_region+'_fire_means'], dates, O3, O3_ymin=0, O3_ymax=0.021)
    Format(ax, dates, O3_color, fire_label, O3_label)
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
    #O3_dict[label+'_means'] = O3_means[sorted_idx]
    O3_dict[label+'_means'] = ButterLowpassFilter(O3_means[sorted_idx], cutoff, 1, order)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    fire_dates = fire_dict[label+'_dates']
    fire_means = fire_dict[label+'_means']

    sorted_dt_list = sorted(enumerate(fire_dates), key=lambda x: x[1])
    sorted_idx = [x[0] for x in sorted_dt_list]
    sorted_idx = np.asarray(sorted_idx)
    sorted_dates = [x[1] for x in sorted_dt_list]

    fire_dict[label+'_dates'] = sorted_dates
    #fire_dict[label+'_means'] = fire_means[sorted_idx]
    fire_dict[label+'_means'] = ButterLowpassFilter(fire_means[sorted_idx], cutoff, 1, order)

    print(label, "dates:", np.shape(fire_dict[label+'_dates']), ", fire:", np.shape(fire_dict[label+'_means']))
    print(label, "dates:", np.shape(O3_dict[label+'_dates']), ", O3:", np.shape(O3_dict[label+'_means']))
#--------------------------------------------------------------------
''' Find intersecting dates and values at those dates '''
print("============================================================")
print("============================================================")
print("============================================================")
intersect_dict = {}
for label in labels:
    #print(np.shape(fire_dict[label+'_dates']), np.shape(fire_dict[label+'_means']))
    print(" - - -", label, "- - -")
    new_dates, data = FindIntersection(x_arrays=[fire_dict[label+'_dates'], O3_dict[label+'_dates']],
                                       y_arrays=[fire_dict[label+'_means'], O3_dict[label+'_means']])
    
    intersect_dict[label+'_fire_means']    = data[0]
    intersect_dict[label+'_O3_means']      = data[1]
    intersect_dict[label+'_dates']         = new_dates

del(fire_dict)
del(O3_dict)
#====================================================================
fig, ax = plt.subplots(2+len(lag_days), 1, figsize=(12, 7), sharex=True)
fig.subplots_adjust(hspace=0, wspace=0)

# fig2, ax2 = plt.subplots(2+len(lag_days), 1, figsize=(12, 7), sharex=True)
# fig2.subplots_adjust(hspace=0, wspace=0)
#--------------------------------------------------------------------
for i in range(len(new_regions)):
    NormalPlot(ax[i], intersect_dict[new_regions[i]+'_dates'], 'blue', fire_region, new_regions[i])

    fire = intersect_dict[fire_region+'_fire_means']
    O3   = intersect_dict[new_regions[i]+'_O3_means']

    corr_string = r'$\rho=$'+str(round(np.corrcoef(fire, O3)[0][1], 3))
    ax[i].text(datetime(2017, 2, 1, 0, 0, 0), .05*max(fire), corr_string)

#--------------------------------------------------------------------
for i in range(len(new_regions),len(new_regions)+len(lag_days)):
    lag = lag_days[i-len(new_regions)]
    #ax[i].plot(new_dates, np.zeros_like(new_dates), color='black', linewidth=1)

    if (lag == 0):
        new_dates = intersect_dict['West_ocean_dates']
        new_O3    = intersect_dict['West_ocean_O3_means'] - intersect_dict['East_ocean_O3_means']
        
        mean_ = np.mean(new_O3)
        corr_string = r'$\rho=$'+str(round(np.corrcoef(intersect_dict[fire_region+'_fire_means'], new_O3)[0][1], 3))

        Plot_O3_Fire(ax[i], new_dates, intersect_dict[fire_region+'_fire_means'], new_dates, new_O3, draw_line_O3=0)
        
        ax[i].text(datetime(2017, 2, 1, 0, 0, 0), .05*max(intersect_dict[fire_region+'_fire_means']), 'West-East\n'+corr_string)
        ax[i].text(max(new_dates), .05*max(intersect_dict[fire_region+'_fire_means']), str(round(mean_, 6))+'\nmore\ndaily\nozone')

        Format(ax[i], new_dates, colors[i-len(new_regions)], fire_region, 'lag='+str(lag)+' days')

    else:
        new_dates = np.asarray(intersect_dict['West_ocean_dates'])
        west_O3 = np.asarray(intersect_dict['West_ocean_O3_means'])
        east_O3 = np.asarray(intersect_dict['East_ocean_O3_means'])
        fire    = np.asarray(intersect_dict[fire_region+'_fire_means'])

        mean_ = np.mean(west_O3[lag:]-east_O3[:-lag])

        corr_string = r'$\rho=$'+str(round(np.corrcoef(fire[lag:], west_O3[lag:]-east_O3[:-lag])[0][1], 3))

        ax[1].plot(new_dates[lag:], east_O3[:-lag], color=colors[i-len(new_regions)], linewidth=1)
        Plot_O3_Fire(ax[i], new_dates[lag:], fire[lag:], new_dates[lag:], west_O3[lag:]-east_O3[:-lag], 
                     O3_color=colors[i-len(new_regions)], draw_line_O3=0)  
        
        ax[i].text(datetime(2017, 2, 1, 0, 0, 0), .05*max(fire), 'West-East\n'+corr_string)
        ax[i].text(max(new_dates), .05*max(fire), str(round(mean_, 6))+'\nmore\ndaily\nozone')
        
        Format(ax[i], new_dates[lag:], colors[i-len(new_regions)], fire_region, 'lag='+str(lag)+' days')
#--------------------------------------------------------------------
    #NormalPlot(ax2[i], intersect_dict[new_regions[i]+'_dates'], 'blue', fire_region, new_regions[i])
fig.savefig(os.path.join("/Users/joshuamiller/Documents/Lancaster/Figs", "Smooth_Fire_O3_lag_"+fire_region+".pdf"), bbox_inches='tight', pad_inches=0)
#====================================================================
plt.show()