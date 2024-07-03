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
from misc.misc_utils import GetDateInStr, FindIntersection, NumericalDerivative
from vis.plotting_utils import ShowYearMonth, Plot_O3_Fire
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

    print(label, "dates:", np.shape(fire_dict[label+'_dates']), ", fire:", np.shape(fire_dict[label+'_means']))
    print(label, "dates:", np.shape(O3_dict[label+'_dates']), ", O3:", np.shape(O3_dict[label+'_means']))
#--------------------------------------------------------------------
''' Compute derivative of fire and ozone '''
print(" - - - - derivative - - - -")
deriv_dict = {}
for label in labels:
    deriv_dates, dfire_dt = NumericalDerivative(fire_dict[label+'_dates'], fire_dict[label+'_means'])
    #print(label, ", dates:",np.shape(deriv_dates), ", fire:", np.shape(dfire_dt))
    deriv_dict[label+'_fire_dt_dates'] = deriv_dates
    deriv_dict[label+'_fire_dt_means'] = dfire_dt

    deriv_dates, dO3_dt = NumericalDerivative(O3_dict[label+'_dates'], O3_dict[label+'_means'])
    #print(label, ", dates:",np.shape(deriv_dates), ", O3:", np.shape(dO3_dt))
    deriv_dict[label+'_O3_dt_dates'] = deriv_dates
    deriv_dict[label+'_O3_dt_means'] = dO3_dt
#--------------------------------------------------------------------
''' Find intersecting dates and values at those dates '''
print("============================================================")
print("============================================================")
print("============================================================")
intersect_dict = {}
for label in labels:
    #print(np.shape(fire_dict[label+'_dates']), np.shape(fire_dict[label+'_means']))
    print(" - - -", label, "- - -")
    new_dates, data = FindIntersection(x_arrays=[fire_dict[label+'_dates'], deriv_dict[label+'_fire_dt_dates'], O3_dict[label+'_dates'], deriv_dict[label+'_O3_dt_dates']],
                                       y_arrays=[fire_dict[label+'_means'], deriv_dict[label+'_fire_dt_means'], O3_dict[label+'_means'], deriv_dict[label+'_O3_dt_means']])
    
    intersect_dict[label+'_fire_means']    = data[0]
    intersect_dict[label+'_fire_dt_means'] = data[1]
    intersect_dict[label+'_O3_means']      = data[2]
    intersect_dict[label+'_O3_dt_means']   = data[3]
    intersect_dict[label+'_dates']         = new_dates

del(fire_dict)
del(O3_dict)
del(deriv_dict)
#====================================================================
fig, ax = plt.subplots(len(labels), 1, figsize=(12, 7), sharex=True)
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
        
fig.savefig(os.path.join("/Users/joshuamiller/Documents/Lancaster/Figs", "Raw_Fire_O3.pdf"), bbox_inches=None, pad_inches=0)
#====================================================================
fig2, ax2 = plt.subplots(len(labels), 1, figsize=(12, 7), sharex=True)
fig2.subplots_adjust(hspace=0, wspace=0)

for i in range(len(labels)):
    dates  = intersect_dict[labels[i]+'_dates']
    fire   = intersect_dict[labels[i]+'_fire_means']
    dO3_dt = intersect_dict[labels[i]+'_O3_dt_means']

    Plot_O3_Fire(ax2[i], dates, fire, dates, dO3_dt, O3_ylabel=r'$\frac{dO_3}{dt}$', draw_line_O3=0)
    
    ax2[i].set_xlim(datetime(2018, 1, 1, 0, 0, 0), datetime(2023, 1, 1, 0, 0, 0))

    ShowYearMonth(ax2[i], dates)

    corr_string = r'$\rho=$'+str(round(np.corrcoef(fire, dO3_dt)[0][1], 3))

    if ('ocean' in labels[i]):
        ax2[i].set_ylim(-0.1, 10)
        ax2[i].text(datetime(2018, 2, 1, 0, 0, 0), 7, labels[i]+'\n'+corr_string)
    else:
        ax2[i].text(datetime(2018, 2, 1, 0, 0, 0), 
                  .7 * max(fire), 
                  labels[i]+'\n'+corr_string)

    #PlotWindow(ax2[i], datetime(2018, 1, 1, 0, 0, 0), datetime(2018, 4, 1, 0, 0, 0), 0, 100, 'lightcoral', .5, 'fire')
    #PlotWindow(ax2[i], datetime(2018, 3, 1, 0, 0, 0), datetime(2018, 6, 1, 0, 0, 0), 0, 100, 'cyan', .5, 'O3')
        
fig2.savefig(os.path.join("/Users/joshuamiller/Documents/Lancaster/Figs", "Raw_Fire_dO3dt.pdf"), bbox_inches=None, pad_inches=0)
#====================================================================
fig3, ax3 = plt.subplots(len(labels), 1, figsize=(12, 7), sharex=True)
fig3.subplots_adjust(hspace=0, wspace=0)

for i in range(len(labels)):
    dates  = intersect_dict[labels[i]+'_dates']
    fire_dt   = intersect_dict[labels[i]+'_fire_dt_means']
    O3 = intersect_dict[labels[i]+'_O3_means']

    Plot_O3_Fire(ax3[i], dates, fire_dt, dates, O3, fire_ylabel=r'$\frac{dFire}{dt}$', draw_line_fire=0)
    
    ax3[i].set_xlim(datetime(2018, 1, 1, 0, 0, 0), datetime(2023, 1, 1, 0, 0, 0))

    ShowYearMonth(ax3[i], dates)

    corr_string = r'$\rho=$'+str(round(np.corrcoef(fire_dt, O3)[0][1], 3))

    if ('ocean' in labels[i]):
        ax3[i].set_ylim(-0.1, 10)
        ax3[i].text(datetime(2018, 2, 1, 0, 0, 0), 7, labels[i]+'\n'+corr_string)
    else:
        ax3[i].text(datetime(2018, 2, 1, 0, 0, 0), 
                  .5 * max(fire_dt), 
                  labels[i]+'\n'+corr_string)

    #PlotWindow(ax3[i], datetime(2018, 1, 1, 0, 0, 0), datetime(2018, 4, 1, 0, 0, 0), 0, 100, 'lightcoral', .5, 'fire')
    #PlotWindow(ax3[i], datetime(2018, 3, 1, 0, 0, 0), datetime(2018, 6, 1, 0, 0, 0), 0, 100, 'cyan', .5, 'O3')
        
fig3.savefig(os.path.join("/Users/joshuamiller/Documents/Lancaster/Figs", "Raw_Firedt_O3.pdf"), bbox_inches=None, pad_inches=0)
#====================================================================
windowsizes = [15, 30, 60, 120]
overlaps    = [0, .5]
colors      = ['springgreen', 'violet', 'royalblue']

fig4, ax4 = plt.subplots(1+len(windowsizes), 1, figsize=(12, 7), sharex=True)
fig4.subplots_adjust(hspace=0, wspace=0)

dates = intersect_dict[labels[region]+'_dates']
fire  = intersect_dict[labels[region]+'_fire_means']
O3    = intersect_dict[labels[region]+'_O3_means']

Plot_O3_Fire(ax4[0], dates, fire, dates, O3)

for i in range(len(windowsizes)):
    ax4[i+1].plot(dates, np.zeros_like(dates), 'k--', alpha=.5)
    for j in range(len(overlaps)):

        offset = int(windowsizes[i] * (1 - overlaps[j]))
        num_corrs = np.shape(dates)[0] - offset - windowsizes[i]
        corrs = np.ones(num_corrs, float) * -9999

        ax4[i+1].axvspan(new_dates[j * 365], new_dates[j * 365]+timedelta(days=windowsizes[i]), 0, 1, color=colors[j], alpha=.1)
        ax4[i+1].axvspan(new_dates[j * 365]+timedelta(days=offset), new_dates[j * 365]+timedelta(days=windowsizes[i])+timedelta(days=offset), 0, 1, color=colors[j], alpha=.1)

        for k in range(num_corrs):
            corrs[k] = np.corrcoef(fire[k:(k+windowsizes[i])], O3[(k+offset):(k+windowsizes[i]+offset)])[0][1]
            #print('w=', windowsizes[i], ', o=', offset)
            #print(k, new_dates[k], new_dates[k+windowsizes[i]],'|', new_dates[k+offset], new_dates[k+windowsizes[i]+offset])

        ax4[i+1].plot(dates[:num_corrs], corrs, color=colors[j], label=str(100*overlaps[j])+'% overlap')        
        ax4[i+1].set_ylim(-1.25, 1.25)
        ax4[i+1].set_ylabel('Correlation')
        ax4[i+1].legend(loc='upper right')
        ax4[i+1].text(datetime(2018, 2, 1, 0, 0, 0), .95, 'Window size: '+str(windowsizes[i])+' days')

    ax4[i+1].set_xlim(datetime(2018, 1, 1, 0, 0, 0), datetime(2023, 1, 1, 0, 0, 0))

ax4[0].set_xlim(datetime(2018, 1, 1, 0, 0, 0), datetime(2023, 1, 1, 0, 0, 0))
ax4[0].set_title(labels[region])
#====================================================================
fig5, ax5 = plt.subplots(1,1,figsize=(12, 7))

corrs = np.ones((len(labels), len(labels)), float) * -999
fire_labels = []
O3_labels = []
for i in range(len(labels)):
    for j in range(len(labels)):
        #print(labels[i]+'fire_means', labels[j]+'O3_means', np.corrcoef(intersect_dict[labels[i]+'_fire_means'], intersect_dict[labels[j]+'_O3_means'])[0][1])
        
        corrs[i][j] = np.corrcoef(intersect_dict[labels[i]+'_fire_means'],
                                  intersect_dict[labels[j]+'_O3_means'])[0][1]
    fire_labels.append(labels[i]+' fire')
    O3_labels.append(labels[i]+' O3')

lol = ax5.imshow(corrs, cmap='bwr', vmin=-1, vmax=1)

divider = make_axes_locatable(ax5)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(lol, cax=cax)
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=180)

ax5.set_xticks(range(len(O3_labels)), O3_labels, rotation=30)
ax5.set_yticks(range(len(fire_labels)), fire_labels)
ax5.set_title('Fire vs. Ozone')
for i in range(np.shape(corrs)[0]):
    for j in range(np.shape(corrs)[1]):
        ax5.text(j, i, f"{corrs[i, j]:.4f}", ha="center", va="center", color="black")

fig5.savefig(os.path.join("/Users/joshuamiller/Documents/Lancaster/Figs", "Raw_Corr_Fire_O3.pdf"), bbox_inches=None, pad_inches=0)
#====================================================================
fig6, ax6 = plt.subplots(1,1,figsize=(12, 7))

corrs = np.ones((len(labels), len(labels)), float) * -999
fire_labels = []
for i in range(len(labels)):
    for j in range(len(labels)):
        #print(labels[i]+'fire_means', labels[j]+'fire_means', np.corrcoef(intersect_dict[labels[i]+'_fire_means'], intersect_dict[labels[j]+'_fire_means'])[0][1])
        
        corrs[i][j] = np.corrcoef(intersect_dict[labels[i]+'_fire_means'],
                                  intersect_dict[labels[j]+'_fire_means'])[0][1]
    fire_labels.append(labels[i]+' fire')

lol = ax6.imshow(corrs, cmap='bwr', vmin=-1, vmax=1)

divider = make_axes_locatable(ax6)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(lol, cax=cax)
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=180)

ax6.set_xticks(range(len(fire_labels)), fire_labels, rotation=30)
ax6.set_yticks(range(len(fire_labels)), fire_labels)
ax6.set_title('Fire vs. Fire')
for i in range(np.shape(corrs)[0]):
    for j in range(np.shape(corrs)[1]):
        ax6.text(j, i, f"{corrs[i, j]:.4f}", ha="center", va="center", color="black")

fig6.savefig(os.path.join("/Users/joshuamiller/Documents/Lancaster/Figs", "Raw_Corr_fire_fire.pdf"), bbox_inches=None, pad_inches=0)
#====================================================================
fig7, ax7 = plt.subplots(1,1,figsize=(12, 7))

corrs = np.ones((len(labels), len(labels)), float) * -999
O3_labels = []
for i in range(len(labels)):
    for j in range(len(labels)):
        #print(labels[i]+'O3_means', labels[j]+'O3_means', np.corrcoef(intersect_dict[labels[i]+'_O3_means'], intersect_dict[labels[j]+'_O3_means'])[0][1])
        
        corrs[i][j] = np.corrcoef(intersect_dict[labels[i]+'_O3_means'],
                                  intersect_dict[labels[j]+'_O3_means'])[0][1]
    O3_labels.append(labels[i]+' O3')

lol = ax7.imshow(corrs, cmap='bwr', vmin=-1, vmax=1)

divider = make_axes_locatable(ax7)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(lol, cax=cax)
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=180)

ax7.set_xticks(range(len(O3_labels)), O3_labels, rotation=30)
ax7.set_yticks(range(len(O3_labels)), O3_labels)
ax7.set_title('Ozone vs. Ozone')
for i in range(np.shape(corrs)[0]):
    for j in range(np.shape(corrs)[1]):
        ax7.text(j, i, f"{corrs[i, j]:.4f}", ha="center", va="center", color="black")

fig7.savefig(os.path.join("/Users/joshuamiller/Documents/Lancaster/Figs", "Raw_Corr_O3_O3.pdf"), bbox_inches=None, pad_inches=0)
#====================================================================
fig8, ax8 = plt.subplots(1,1,figsize=(12, 7))

corrs = np.ones((len(labels), len(labels)), float) * -999
O3_labels = []
fire_dt_labels = []
for i in range(len(labels)):
    for j in range(len(labels)):
        #print(labels[i]+'O3_means', labels[j]+'O3_means', np.corrcoef(intersect_dict[labels[i]+'_O3_means'], intersect_dict[labels[j]+'_O3_means'])[0][1])
        
        corrs[i][j] = np.corrcoef(intersect_dict[labels[i]+'_fire_dt_means'],
                                  intersect_dict[labels[j]+'_O3_means'])[0][1]
    fire_dt_labels.append(labels[i]+' '+r'$\frac{dFire}{dt}$')
    O3_labels.append(labels[i]+' O3')

lol = ax8.imshow(corrs, cmap='bwr', vmin=-1, vmax=1)

divider = make_axes_locatable(ax8)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(lol, cax=cax)
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=180)

ax8.set_xticks(range(len(O3_labels)), O3_labels, rotation=30)
ax8.set_yticks(range(len(fire_dt_labels)), fire_dt_labels)
ax8.set_title(r'$\frac{dFire}{dt}$'+' vs. Ozone')
for i in range(np.shape(corrs)[0]):
    for j in range(np.shape(corrs)[1]):
        ax8.text(j, i, f"{corrs[i, j]:.4f}", ha="center", va="center", color="black")

fig8.savefig(os.path.join("/Users/joshuamiller/Documents/Lancaster/Figs", "Raw_Corr_dFiredt_O3.pdf"), bbox_inches=None, pad_inches=0)
#====================================================================
fig9, ax9 = plt.subplots(1,1,figsize=(12, 7))

corrs = np.ones((len(labels), len(labels)), float) * -999
O3_dt_labels = []
fire_labels = []
for i in range(len(labels)):
    for j in range(len(labels)):
        #print(labels[i]+'O3_means', labels[j]+'O3_means', np.corrcoef(intersect_dict[labels[i]+'_O3_means'], intersect_dict[labels[j]+'_O3_means'])[0][1])
        
        corrs[i][j] = np.corrcoef(intersect_dict[labels[i]+'_fire_means'],
                                  intersect_dict[labels[j]+'_O3_dt_means'])[0][1]
    fire_labels.append(labels[i]+' fire')
    O3_dt_labels.append(labels[i]+' '+r'$\frac{O_3}{dt}$')

lol = ax9.imshow(corrs, cmap='bwr', vmin=-1, vmax=1)

divider = make_axes_locatable(ax9)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(lol, cax=cax)
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=180)

ax9.set_xticks(range(len(O3_dt_labels)), O3_dt_labels, rotation=30)
ax9.set_yticks(range(len(fire_labels)), fire_labels)
ax9.set_title('Fire vs. '+r'$\frac{O_3}{dt}$')
for i in range(np.shape(corrs)[0]):
    for j in range(np.shape(corrs)[1]):
        ax9.text(j, i, f"{corrs[i, j]:.4f}", ha="center", va="center", color="black")

fig9.savefig(os.path.join("/Users/joshuamiller/Documents/Lancaster/Figs", "Raw_Corr_O3_dFiredt.pdf"), bbox_inches=None, pad_inches=0)
#====================================================================
fig10, ax10 = plt.subplots(1,1,figsize=(12, 7))

corrs = np.ones((len(labels), len(labels)), float) * -999
fire_dt_labels = []
O3_dt_labels = []
for i in range(len(labels)):
    for j in range(len(labels)):
        #print(labels[i]+'O3_means', labels[j]+'O3_means', np.corrcoef(intersect_dict[labels[i]+'_O3_means'], intersect_dict[labels[j]+'_O3_means'])[0][1])
        
        corrs[i][j] = np.corrcoef(intersect_dict[labels[i]+'_fire_dt_means'],
                                  intersect_dict[labels[j]+'_O3_dt_means'])[0][1]
    fire_dt_labels.append(labels[i]+' '+r'$\frac{dFire}{dt}$')
    O3_dt_labels.append(labels[i]+' '+r'$\frac{O_3}{dt}$')

lol = ax10.imshow(corrs, cmap='bwr', vmin=-1, vmax=1)

divider = make_axes_locatable(ax10)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(lol, cax=cax)
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=180)

ax10.set_xticks(range(len(O3_dt_labels)), O3_dt_labels, rotation=30)
ax10.set_yticks(range(len(fire_dt_labels)), fire_dt_labels)
ax10.set_title(r'$\frac{dFire}{dt}$'+' vs. '+r'$\frac{dO_3}{dt}$')
for i in range(np.shape(corrs)[0]):
    for j in range(np.shape(corrs)[1]):
        ax10.text(j, i, f"{corrs[i, j]:.4f}", ha="center", va="center", color="black")

fig10.savefig(os.path.join("/Users/joshuamiller/Documents/Lancaster/Figs", "Raw_Corr_dFiredt_dO3dt.pdf"), bbox_inches=None, pad_inches=0)      
#====================================================================
plt.show()