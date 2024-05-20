import sys
sys.dont_write_bytecode = True
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
from datetime import datetime, timedelta
    
sys.path.append(os.getcwd())
from data_utils.extraction_funcs import Extract_netCDF4
from misc.misc_utils import GetDateInStr, FindIntersection
from misc.plotting_utils import ShowYearMonth, Plot_O3_Fire
#====================================================================
base_path    = "/Users/joshuamiller/Documents/Lancaster/Data"
O3_folders   = ['Kriged_L2_O3_TCL', 'East_Ocean/Ozone', 'West_Ocean/Ozone', 'South_Land/Ozone', 'North_Land/Ozone']
fire_folders = ['Kriged_MODIS_C61', 'East_Ocean/Fire', 'West_Ocean/Fire', 'South_Land/Fire', 'North_Land/Fire']
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
#--------------------------------------------------------------------
intersect_dict = {}
for label in labels:
    new_dates, data = FindIntersection(x_arrays=[fire_dict[label+'_dates'], O3_dict[label+'_dates']],
                                       y_arrays=[fire_dict[label+'_means'], O3_dict[label+'_means']])
    new_fire = data[0]
    new_O3   = data[1]

    intersect_dict[label+'_fire_means'] = new_fire
    intersect_dict[label+'_O3_means'] = new_O3
    intersect_dict[label+'_dates'] = new_dates
#====================================================================
fig, ax = plt.subplots(5, 1, figsize=(12, 7), sharex=True)
fig.subplots_adjust(hspace=0, wspace=0)

for i in range(len(labels)):
    dates = intersect_dict[labels[i]+'_dates']
    fire  = intersect_dict[labels[i]+'_fire_means']
    O3    = intersect_dict[labels[i]+'_O3_means']

    Plot_O3_Fire(ax[i], dates, fire, dates, O3)
    
    ax[i].set_xlim(datetime(2018, 1, 1, 0, 0, 0), datetime(2023, 1, 1, 0, 0, 0))

    ShowYearMonth(ax[i], dates)

    corr_string = r'$\rho=$'+str(round(np.corrcoef(fire, O3)[0][1], 3))

    if ('ocean' in labels[i]):
        ax[i].set_ylim(-0.1, 10)
        ax[i].text(datetime(2018, 2, 1, 0, 0, 0), 7, labels[i]+'\n'+corr_string)
    else:
        ax[i].text(datetime(2018, 2, 1, 0, 0, 0), 
                  .7 * max(fire), 
                  labels[i]+'\n'+corr_string)

    #PlotWindow(ax[i], datetime(2018, 1, 1, 0, 0, 0), datetime(2018, 4, 1, 0, 0, 0), 0, 100, 'lightcoral', .5, 'fire')
    #PlotWindow(ax[i], datetime(2018, 3, 1, 0, 0, 0), datetime(2018, 6, 1, 0, 0, 0), 0, 100, 'cyan', .5, 'O3')
        
#fig.savefig(os.path.join("/Users/joshuamiller/Documents/Lancaster/Figs", "RawData_Fire_O3.pdf"), bbox_inches=None, pad_inches=0)
#====================================================================
windowsizes = [15, 30, 60, 120]
overlaps    = [0, .5]
colors      = ['springgreen', 'violet', 'royalblue']

fig2, ax2 = plt.subplots(1+len(windowsizes), 1, figsize=(12, 7), sharex=True)
fig2.subplots_adjust(hspace=0, wspace=0)

dates = intersect_dict[labels[region]+'_dates']
fire  = intersect_dict[labels[region]+'_fire_means']
O3    = intersect_dict[labels[region]+'_O3_means']

Plot_O3_Fire(ax2[0], dates, fire, dates, O3)

for i in range(len(windowsizes)):
    ax2[i+1].plot(dates, np.zeros_like(dates), 'k--', alpha=.5)
    for j in range(len(overlaps)):

        offset = int(windowsizes[i] * (1 - overlaps[j]))
        num_corrs = np.shape(dates)[0] - offset - windowsizes[i]
        corrs = np.ones(num_corrs, float) * -9999

        ax2[i+1].axvspan(new_dates[j * 365], new_dates[j * 365]+timedelta(days=windowsizes[i]), 0, 1, color=colors[j], alpha=.1)
        ax2[i+1].axvspan(new_dates[j * 365]+timedelta(days=offset), new_dates[j * 365]+timedelta(days=windowsizes[i])+timedelta(days=offset), 0, 1, color=colors[j], alpha=.1)

        for k in range(num_corrs):
            corrs[k] = np.corrcoef(new_fire[k:(k+windowsizes[i])], new_O3[(k+offset):(k+windowsizes[i]+offset)])[0][1]
            #print('w=', windowsizes[i], ', o=', offset)
            #print(k, new_dates[k], new_dates[k+windowsizes[i]],'|', new_dates[k+offset], new_dates[k+windowsizes[i]+offset])

        ax2[i+1].plot(dates[:num_corrs], corrs, color=colors[j], label=str(100*overlaps[j])+'% overlap')        
        ax2[i+1].set_ylim(-1.25, 1.25)
        ax2[i+1].set_ylabel('Correlation')
        ax2[i+1].legend(loc='upper right')
        ax2[i+1].text(datetime(2018, 2, 1, 0, 0, 0), .95, 'Window size: '+str(windowsizes[i])+' days')

    ax2[i+1].set_xlim(datetime(2018, 1, 1, 0, 0, 0), datetime(2023, 1, 1, 0, 0, 0))

ax2[0].set_xlim(datetime(2018, 1, 1, 0, 0, 0), datetime(2023, 1, 1, 0, 0, 0))
ax2[0].set_title(labels[region])
#====================================================================
fig3, ax3 = plt.subplots(1,1,figsize=(12, 7))

corrs = np.ones((len(labels), len(labels)), float) * -999
fire_labels = []
O3_labels = []
for i in range(len(labels)):
    for j in range(len(labels)):
        print(labels[i]+'fire_means', labels[j]+'O3_means', np.corrcoef(intersect_dict[labels[i]+'_fire_means'], intersect_dict[labels[j]+'_O3_means'])[0][1])
        
        corrs[i][j] = np.corrcoef(intersect_dict[labels[i]+'_fire_means'],
                                  intersect_dict[labels[j]+'_O3_means'])[0][1]
    fire_labels.append(labels[i]+' fire')
    O3_labels.append(labels[i]+' O3')

lol = ax3.imshow(corrs, cmap='bwr', vmin=-1, vmax=1)

divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(lol, cax=cax)
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=180)

ax3.set_xticks(range(len(labels)), O3_labels, rotation=30)
ax3.set_yticks(range(len(labels)), fire_labels)
ax3.set_title('Fire vs. Ozone')
for i in range(np.shape(corrs)[0]):
    for j in range(np.shape(corrs)[1]):
        ax3.text(j, i, f"{corrs[i, j]:.4f}", ha="center", va="center", color="black")

#fig3.savefig(os.path.join("/Users/joshuamiller/Documents/Lancaster/Figs", "Corr_Fire_O3.pdf"), bbox_inches=None, pad_inches=0)
#====================================================================
fig4, ax4 = plt.subplots(1,1,figsize=(12, 7))

corrs = np.ones((len(labels), len(labels)), float) * -999
fire_labels = []
for i in range(len(labels)):
    for j in range(len(labels)):
        print(labels[i]+'fire_means', labels[j]+'fire_means', np.corrcoef(intersect_dict[labels[i]+'_fire_means'], intersect_dict[labels[j]+'_fire_means'])[0][1])
        
        corrs[i][j] = np.corrcoef(intersect_dict[labels[i]+'_fire_means'],
                                  intersect_dict[labels[j]+'_fire_means'])[0][1]
    fire_labels.append(labels[i]+' fire')

lol = ax4.imshow(corrs, cmap='bwr', vmin=-1, vmax=1)

divider = make_axes_locatable(ax4)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(lol, cax=cax)
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=180)

ax4.set_xticks(range(len(labels)), fire_labels, rotation=30)
ax4.set_yticks(range(len(labels)), fire_labels)
ax4.set_title('Fire vs. Fire')
for i in range(np.shape(corrs)[0]):
    for j in range(np.shape(corrs)[1]):
        ax4.text(j, i, f"{corrs[i, j]:.4f}", ha="center", va="center", color="black")

#fig4.savefig(os.path.join("/Users/joshuamiller/Documents/Lancaster/Figs", "Corr_fire_fire.pdf"), bbox_inches=None, pad_inches=0)
#====================================================================
fig5, ax5 = plt.subplots(1,1,figsize=(12, 7))

corrs = np.ones((len(labels), len(labels)), float) * -999
O3_labels = []
for i in range(len(labels)):
    for j in range(len(labels)):
        print(labels[i]+'O3_means', labels[j]+'O3_means', np.corrcoef(intersect_dict[labels[i]+'_O3_means'], intersect_dict[labels[j]+'_O3_means'])[0][1])
        
        corrs[i][j] = np.corrcoef(intersect_dict[labels[i]+'_O3_means'],
                                  intersect_dict[labels[j]+'_O3_means'])[0][1]
    O3_labels.append(labels[i]+' O3')

lol = ax5.imshow(corrs, cmap='bwr', vmin=-1, vmax=1)

divider = make_axes_locatable(ax5)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(lol, cax=cax)
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=180)

ax5.set_xticks(range(len(labels)), O3_labels, rotation=30)
ax5.set_yticks(range(len(labels)), O3_labels)
ax5.set_title('Ozone vs. Ozone')
for i in range(np.shape(corrs)[0]):
    for j in range(np.shape(corrs)[1]):
        ax5.text(j, i, f"{corrs[i, j]:.4f}", ha="center", va="center", color="black")

#fig5.savefig(os.path.join("/Users/joshuamiller/Documents/Lancaster/Figs", "Corr_O3_O3.pdf"), bbox_inches=None, pad_inches=0)
plt.show()