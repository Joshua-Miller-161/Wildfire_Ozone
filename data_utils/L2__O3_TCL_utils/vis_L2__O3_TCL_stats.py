import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.dates as mdates
import geopandas as gpd
from matplotlib.colors import Normalize, LogNorm, BoundaryNorm, TwoSlopeNorm
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
import numpy as np
import sys
from statsmodels.tsa.stattools import adfuller

sys.path.append(os.getcwd())
from data_utils.preprocessing_funcs import Scale
from data_utils.extraction_funcs import Extract_netCDF4
#====================================================================
''' Make subplot '''
fig, ax = plt.subplots(3, 2, figsize=(10, 8))
fig.subplots_adjust(wspace=.5, hspace=.5)

replace_val = 0.001

mode = 'ct'
#====================================================================
''' Get data '''
folder_filtered = "/Users/joshuamiller/Documents/Lancaster/Data/Filtered_L2_O3_TCL"
folder_kriged   = "/Users/joshuamiller/Documents/Lancaster/Data/Kriged_L2_O3_TCL"
#--------------------------------------------------------------------
filtered_dates = []
files_filtered = os.listdir(folder_filtered)
files_filtered.sort()
for file_filtered in files_filtered:
    if (file_filtered.endswith('nc')):
        dict_ = Extract_netCDF4(os.path.join(folder_filtered, file_filtered),
                                    ['dates_for_tropospheric_column'],
                                    groups='all',
                                    print_sum=False)
        start_date = list(''.join(dict_['dates_for_tropospheric_column']).split(' '))[0]
        filtered_dates.append(start_date)

filtered_dates.sort()
#print(filtered_dates)
lat_for_plot = 0
lon_for_plot = 0
for file_filtered in files_filtered:
    if (file_filtered.endswith('nc')):
        init_dict = Extract_netCDF4(os.path.join(folder_filtered, file_filtered),
                                    ['lat', 'lon', 'ozone_tropospheric_vertical_column', 'dates_for_tropospheric_column'],
                                    groups='all',
                                    print_sum=False)
        lat_for_plot = init_dict['lat']
        lon_for_plot = init_dict['lon']
        break

filtered_O3 = np.ones((len(filtered_dates), np.shape(lat_for_plot)[0], np.shape(lon_for_plot)[0]), float) * -0.01
filtered_lat = np.tile(init_dict['lat'], (np.shape(init_dict['lon'])[0], 1)).T
filtered_lon = np.tile(init_dict['lon'], (np.shape(init_dict['lat'])[0], 1))
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
for file_filtered in files_filtered:
    if file_filtered.endswith('.nc'):
        dict_ = Extract_netCDF4(os.path.join(folder_filtered, file_filtered),
                                ['lat', 'lon', 'ozone_tropospheric_vertical_column', 'dates_for_tropospheric_column'],
                                groups='all',
                                print_sum=False)

        lat = dict_['lat']
        lon = dict_['lon']
        ozone = dict_['ozone_tropospheric_vertical_column']
        start_date = list(''.join(dict_['dates_for_tropospheric_column']).split(' '))[0]
        date_idx = [i for i, x in enumerate(filtered_dates) if x == start_date]
        #print(" = + = + = date_idx =", date_idx)
        for idx in date_idx:
            filtered_O3[idx][:][:] = ozone
#--------------------------------------------------------------------
kriged_dates = []
files_kriged = os.listdir(folder_kriged)
files_kriged.sort()
for file_kriged in files_kriged:
    if (file_kriged.endswith('nc')):
        dict_ = Extract_netCDF4(os.path.join(folder_kriged, file_kriged),
                                    ['dates_for_tropospheric_column'],
                                    groups='all',
                                    print_sum=False)
        start_date = list(''.join(dict_['dates_for_tropospheric_column']).split(' '))[0]
        kriged_dates.append(start_date)

kriged_dates.sort()
#print(kriged_dates)
lat_for_plot = 0
lon_for_plot = 0
for file_kriged in files_kriged:
    if (file_kriged.endswith('nc')):
        init_dict = Extract_netCDF4(os.path.join(folder_kriged, file_kriged),
                                    ['lat', 'lon', 'ozone_tropospheric_vertical_column', 'dates_for_tropospheric_column'],
                                    groups='all',
                                    print_sum=False)
        lat_for_plot = init_dict['lat']
        lon_for_plot = init_dict['lon']
        break

kriged_O3 = np.ones((len(kriged_dates), np.shape(lat_for_plot)[0], np.shape(lon_for_plot)[0]), float) * -0.01
kriged_lat = np.tile(init_dict['lat'], (np.shape(init_dict['lon'])[0], 1)).T
kriged_lon = np.tile(init_dict['lon'], (np.shape(init_dict['lat'])[0], 1))

print("OSIDSDOISODIF_#R(#(RWDOFJWE)F*UDSOIFNSDPOFU()WDSFUSODNF)")
print(np.shape(filtered_lon), np.shape(filtered_lat), min(filtered_lon.ravel()), max(filtered_lon.ravel()), min(filtered_lat.ravel()), max(filtered_lat.ravel()))
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
for file_kriged in files_kriged:
    if file_kriged.endswith('.nc'):
        dict_ = Extract_netCDF4(os.path.join(folder_kriged, file_kriged),
                                ['lat', 'lon', 'ozone_tropospheric_vertical_column', 'dates_for_tropospheric_column'],
                                groups='all',
                                print_sum=False)

        lat = dict_['lat']
        lon = dict_['lon']
        ozone = dict_['ozone_tropospheric_vertical_column']
        start_date = list(''.join(dict_['dates_for_tropospheric_column']).split(' '))[0]
        date_idx = [i for i, x in enumerate(kriged_dates) if x == start_date]
        #print(" = + = + = date_idx =", date_idx)
        for idx in date_idx:
            kriged_O3[idx][:][:] = ozone
#====================================================================
''' Calculations '''          
filtered_datetimes = [datetime.strptime(d, '%Y%m%d') for d in filtered_dates]
filtered_O3_mean   = np.nanmean(filtered_O3, axis=(1,2))

kriged_datetimes = [datetime.strptime(d, '%Y%m%d') for d in kriged_dates]
kriged_O3_mean   = np.mean(kriged_O3, axis=(1,2))

diff = kriged_O3_mean - filtered_O3_mean

# - - - - - - - - - Augmented Dickey-Fuller - - - - - - - - - - - - -
# "Note: If P-Value is smaller than 0.05, we reject the null hypothesis and the series is stationary"
adftest = adfuller(filtered_O3_mean, autolag='AIC', regression=mode)
filtered_O3_mean_stat = adftest[0]
filtered_O3_mean_pval = adftest[1]

adftest = adfuller(kriged_O3_mean, autolag='AIC', regression=mode)
kriged_O3_mean_stat = adftest[0]
kriged_O3_mean_pval = adftest[1]

filtered_adf_stats = np.ones((np.shape(filtered_O3)[1], np.shape(filtered_O3)[2]), float) * 99
filtered_pvals     = np.ones_like(filtered_adf_stats, float) * 99
for i in range(np.shape(filtered_O3)[1]): # lat
    for j in range(np.shape(filtered_O3)[2]): # lon
        arr = filtered_O3[:, i, j]
        mask = np.isnan(arr)
        mask = np.logical_not(mask)
        filtered_arr = arr[mask]

        adftest = adfuller(filtered_arr, autolag='AIC', regression=mode)
        filtered_adf_stats[i, j] = adftest[0]
        filtered_pvals[i, j] = adftest[1]
        print("i=", i, ", j=", j, ", filtered stat:", round(adftest[0], 4), ", filtered pval:", round(adftest[1], 4))

kriged_adf_stats = np.ones((np.shape(kriged_O3)[1], np.shape(kriged_O3)[2]), float) * 99
kriged_pvals     = np.ones_like(kriged_adf_stats, float) * 99
for i in range(np.shape(kriged_O3)[1]): # lat
    for j in range(np.shape(kriged_O3)[2]): # lon
        adftest = adfuller(kriged_O3[:, i, j], autolag='AIC', regression=mode)
        kriged_adf_stats[i, j] = adftest[0]
        kriged_pvals[i, j] = adftest[1]
        print("i=", i, ", j=", j, ", kriged stat:", round(adftest[0], 4), ", kriged pval:", round(adftest[1], 4))
#====================================================================
''' Make colomaps '''
stat_cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'deepskyblue', 'lavenderblush', 'pink', 'red'], N=800)
stat_norm = Normalize(min([min(filtered_adf_stats.ravel()), min(kriged_adf_stats.ravel())]),
                      max([max(filtered_adf_stats.ravel()), max(kriged_adf_stats.ravel())]))

# colors = ["blue", "green", "yellow", "red"]
# boundaries = [min([min(filtered_pvals.ravel()), min(kriged_pvals.ravel())]), 
#               0.05,
#               max([max(filtered_pvals.ravel()), max(kriged_pvals.ravel())])]

# pval_cmap = 'RdBu_r'
# pval_norm = BoundaryNorm(boundaries, pval_cmap.N)

pval_norm = TwoSlopeNorm(vmin=min([min(filtered_pvals.ravel()), min(kriged_pvals.ravel())]), 
                        vcenter=0.05, 
                        vmax=max([max(filtered_pvals.ravel()), max(kriged_pvals.ravel())]))
pval_cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'deepskyblue', 'lavenderblush', 'pink', 'red'], N=800)
#====================================================================
''' Plot '''
ax[0][0].plot(filtered_datetimes, filtered_O3_mean, color='dodgerblue', linewidth=1, label='Raw data')
ax[0][0].plot(kriged_datetimes, kriged_O3_mean, color='hotpink', linewidth=1, label='Interpolated data')

ax[0][1].scatter(filtered_datetimes, diff, 
                 s=1, facecolors=None, edgecolors='mediumvioletred', label='Interp. - raw')
ax[0][1].plot(filtered_datetimes, np.zeros_like(filtered_datetimes), 'k--', linewidth=1)

# - - - - - - - - - Augmented Dickey-Fuller - - - - - - - - - - -
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(np.shape(filtered_lon), np.shape(filtered_lat), np.shape(filtered_pvals),
      min(filtered_lon.ravel()), max(filtered_lon.ravel()), min(filtered_lat.ravel()), max(filtered_lat.ravel()))
filtered_pval_plot = ax[1][0].scatter(x=filtered_lon, y=filtered_lat, c=filtered_pvals,
                     s=2, marker='s', norm=pval_norm, cmap=pval_cmap)
divider = make_axes_locatable(ax[1][0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(filtered_pval_plot, cax=cax, label=r'p-val')

kriged_pval_plot = ax[1][1].scatter(x=kriged_lon, y=kriged_lat, c=kriged_pvals,
                   s=2, marker='s', norm=pval_norm, cmap=pval_cmap)
divider = make_axes_locatable(ax[1][1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(kriged_pval_plot, cax=cax, label=r'p-val')


filtered_stat_plot = ax[2][0].scatter(x=filtered_lon, y=filtered_lat, c=filtered_adf_stats,
                     s=2, marker='s', norm=stat_norm, cmap=stat_cmap)
divider = make_axes_locatable(ax[2][0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(filtered_stat_plot, cax=cax, label=r'statistic')

kriged_stat_plot = ax[2][1].scatter(x=kriged_lon, y=kriged_lat, c=kriged_adf_stats,
                   s=2, marker='s', norm=stat_norm, cmap=stat_cmap)
divider = make_axes_locatable(ax[2][1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(kriged_stat_plot, cax=cax, label=r'statistic')
#====================================================================
''' Plot world map '''
world = gpd.read_file("/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")

#====================================================================
''' Format '''
titles = ['Raw data | ', "Interpolated data | ", "Raw - Interpolated | "]

ax[0][0].set_title('Daily mean over bounding box\nADF (filtered): p='+str(round(filtered_O3_mean_pval, 5))+', stat='+str(round(filtered_O3_mean_stat, 5))+'\nADF (interpolated): p='+str(round(kriged_O3_mean_pval, 5))+', stat='+str(round(kriged_O3_mean_stat, 5)))

ax[0][1].set_title('Residuals')
ax[0][1].set_ylim(-max(abs(diff))-0.0001, max(abs(diff))+0.0001)
mask = (diff < 0) & (~np.isnan(diff))
ax[0][1].text(filtered_datetimes[0], -max(abs(diff)), '% < 0: '+str(round(100 * np.sum(mask) / np.shape(diff)[0], 3)))
mask = (diff > 0) & (~np.isnan(diff))
ax[0][1].text(filtered_datetimes[0], max(abs(diff))-0.0001, '% > 0: '+str(round(100 * np.sum(mask) / np.shape(diff)[0], 3)))

for i in range(2):
    ax[0][i].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax[0][i].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax[0][i].tick_params(axis='x', rotation=45)
    ax[0][i].legend()
    ax[1][i].set_title(titles[i]+'ADF p-value')
    world.plot(ax=ax[1][i], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
    ax[1][i].set_xlim(-20, 60)
    ax[1][i].set_ylim(-20, 20)
    ax[2][i].set_title(titles[i]+'ADF statistic')
    world.plot(ax=ax[2][i], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
    ax[2][i].set_xlim(-20, 60)
    ax[2][i].set_ylim(-20, 20)
#====================================================================
plt.show()
fig.savefig(os.path.join('/Users/joshuamiller/Documents/Lancaster/Figs', "L2__O3_TCL_stats_"+mode+".pdf"), bbox_inches = 'tight', pad_inches = 0)