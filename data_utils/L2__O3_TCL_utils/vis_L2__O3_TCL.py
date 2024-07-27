import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
from matplotlib.colors import Normalize, LogNorm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
from shapely.geometry import Point
import netCDF4 as nc
import numpy as np
import sys

sys.path.append(os.getcwd())
from data_utils.preprocessing_funcs import Scale
from data_utils.extract_netCDF4 import Extract_netCDF4
from vis.plotting_utils import PlotBoxes, ShowYearMonth, DegreeFormatter
#====================================================================
font = {'weight' : 'bold',
        'size'   : 11}

plt.rc('font', **font)
#====================================================================
''' Make subplot '''
fig, ax = plt.subplots(3, 1, figsize=(8, 6))
fig.subplots_adjust(wspace=0, hspace=.5)

replace_val = 0.001
#====================================================================
print("============================================================")
print("============================================================")
print("============================================================")
print("============================================================")
print("============================================================")
print("============================================================")

''' Get data '''
folder_0 = "/Users/joshuamiller/Documents/Lancaster/Data/L2_O3_TCL"
f = "/Users/joshuamiller/Documents/Lancaster/Data/L2_O3_TCL/S5P_RPRO_L2__O3_TCL_20200725T121326_20200731T125843_14459_03_020401_20230329T125441/S5P_RPRO_L2__O3_TCL_20200725T121326_20200731T125843_14459_03_020401_20230329T125441.nc"

dict_ = Extract_netCDF4(os.path.join(folder_0, f),
                        ['latitude_ccd', 'longitude_ccd', 'ozone_tropospheric_vertical_column', 'dates_for_tropospheric_column'],
                        groups='all',
                        print_sum=False)

lat_0 = dict_['latitude_ccd']
lon_0 = dict_['longitude_ccd']
ozone_0 = np.squeeze(dict_['ozone_tropospheric_vertical_column'])
ozone_0 = np.ma.getdata(np.squeeze(dict_['ozone_tropospheric_vertical_column']))
ozone_0 = np.where(ozone_0 > 100, replace_val, ozone_0)
ozone_0 = np.where(ozone_0 <= 0, 10**-10, ozone_0)

#print(np.shape(lon_0), np.shape(lat_0), np.shape(ozone_0))
#print("lon:", lon_0[0:3], ", lat:", lat_0[0:3], ", o3:", ozone_0[0:3, 0:3])

dates_0 = ''.join(dict_['dates_for_tropospheric_column'])
#--------------------------------------------------------------------
folder_1 = "/Users/joshuamiller/Documents/Lancaster/Data/Filtered_L2_O3_TCL"
for file_1 in os.listdir(folder_1):
    if file_1.endswith('.nc'):
        dataset = nc.Dataset(os.path.join(folder_1, file_1), 'r')
        if (f[-30:-1] in dataset.source):
            print("---")
            print("MATCH: dates_0: ", dates_0, "\n", dataset.source)
            print(type(folder_1), type(file_1), file_1)
            print("---")
            break

dict_ = Extract_netCDF4(os.path.join(folder_1, file_1),
                        ['lat', 'lon', 'ozone_tropospheric_vertical_column', 'dates_for_tropospheric_column'],
                        groups='all',
                        print_sum=False)

lat_1 = dict_['lat']
lon_1 = dict_['lon']
ozone_1 = np.squeeze(dict_['ozone_tropospheric_vertical_column'])
ozone_1 = np.ma.getdata(np.squeeze(dict_['ozone_tropospheric_vertical_column']))
ozone_1 = np.where(ozone_1 > 100, replace_val, ozone_1)
ozone_1 = np.nan_to_num(ozone_1, nan=replace_val)
ozone_1 = np.where(ozone_1 <= 0, 10**-10, ozone_1)

#print(np.shape(lon_1), np.shape(lat_1), np.shape(ozone_1))
#print("lon:", lon_1[0:3], ", lat:", lat_1[0:3], ", o3:", ozone_1[0:3, 0:3])
#print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
dates_1 = ''.join(dict_['dates_for_tropospheric_column'])

#--------------------------------------------------------------------
folder_3 = "/Users/joshuamiller/Documents/Lancaster/Data/Kriged_L2_O3_TCL"
for file_3 in os.listdir(folder_3):
    if file_3.endswith('.nc'):
        dataset = nc.Dataset(os.path.join(folder_3, file_3), 'r')
        if (file_1 in dataset.source):
            print("---")
            print("MATCH: dates_0: ", dates_0, "\n", dataset.source)
            print("---")
            break
        
dict_ = Extract_netCDF4(os.path.join(folder_3, file_3),
                        ['lat', 'lon', 'ozone_tropospheric_vertical_column', 'dates_for_tropospheric_column'],
                        groups='all',
                        print_sum=False)

lat_3 = dict_['lat']
lon_3 = dict_['lon']
ozone_3 = np.squeeze(dict_['ozone_tropospheric_vertical_column'])
dates_3 = ''.join(dict_['dates_for_tropospheric_column'])

#====================================================================
''' Plot ozone '''
date = 0

# - - - - - - - - - Get points for the ozone plot - - - - - - - - - - -
lat_tiled_0 = np.tile(lat_0, (np.shape(lon_0)[0], 1)).T
lon_tiled_0 = np.tile(lon_0, (np.shape(lat_0)[0], 1))
# print("00000000000")
# print("lat_tiled:", lat_tiled_0)
# print('00000000000')
# print("lon_tiled:", lon_tiled_0)
# print('00000000000')
lat_tiled_1 = np.tile(lat_1, (np.shape(lon_1)[0], 1)).T
lon_tiled_1 = np.tile(lon_1, (np.shape(lat_1)[0], 1))

latlon_points = np.array(list(zip(lat_tiled_1.ravel(), lon_tiled_1.ravel())))
O3_points = ozone_1.ravel()
print('00000000000')
print(np.shape(latlon_points), np.shape(O3_points), latlon_points[:5,:])



lat_tiled_3 = np.tile(lat_3, (np.shape(lon_3)[0], 1)).T
lon_tiled_3 = np.tile(lon_3, (np.shape(lat_3)[0], 1))
# - - - - - - - - - - - Make colorbar for ozone - - - - - - - - - - -
print(min(ozone_1.ravel()), min(ozone_0.ravel()))
# ozone_norm = Normalize(vmin=min([min(ozone_1.ravel()), min(ozone_0.ravel())]), 
#                        vmax=max([max(ozone_1.ravel()), max(ozone_0.ravel())]))
ozone_norm = Normalize(vmin=0, vmax=0.024)
ozone_cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'deepskyblue', 'lavenderblush', 'pink', 'red'], N=800) #8 Higher N=more smooth

# - - - - - - - - - - - - - Plot ozone data - - - - - - - - - - - - -
O3_0 = ax[0].scatter(x=lon_tiled_0, y=lat_tiled_0, c=ozone_0, 
                     s=2, marker='o', norm=ozone_norm, cmap=ozone_cmap)

# for i in range(np.shape(ozone_0)[0]): # Latitude dimension
#     for j in range(np.shape(ozone_0)[1]): # Longitude dimension
#         O3_0 = ax[0].scatter(x=lon_0[j], y=lat_0[i], c=ozone_0[i][j], 
#                              s=2, marker='o', norm=ozone_norm, cmap=ozone_cmap)

divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(O3_0, cax=cax, label=r'$\frac{mol}{m^2}$')



O3_1 = ax[1].scatter(x=lon_tiled_1, y=lat_tiled_1, c=ozone_1,
                     s=2, marker='o', norm=ozone_norm, cmap=ozone_cmap)
# for i in range(np.shape(ozone_1)[0]): # Latitude dimension
#     for j in range(np.shape(ozone_1)[1]): # Longitude dimension
#         O3_1 = ax[1].scatter(x=lon_1[j], y=lat_1[i], c=ozone_1[i][j], 
#                              s=2, marker='o', norm=ozone_norm, cmap=ozone_cmap)
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(O3_1, cax=cax, label=r'$\frac{mol}{m^2}$')



# O3_points = ax[2].scatter(x=latlon_points[:, 1], y=latlon_points[:, 0], c=O3_points,
#                           s=2, marker='o', norm=ozone_norm, cmap=ozone_cmap)

# divider = make_axes_locatable(ax[2])
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(O3_points, cax=cax, label=r'$\frac{mol}{m^2}$')


O3_3 = ax[2].scatter(x=lon_tiled_3, y=lat_tiled_3, c=ozone_3,
                     s=2, marker='o', norm=ozone_norm, cmap=ozone_cmap)

divider = make_axes_locatable(ax[2])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(O3_3, cax=cax, label=r'$\frac{mol}{m^2}$')
#====================================================================
''' Plot world map '''
world = gpd.read_file("/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")

world.plot(ax=ax[0], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
world.plot(ax=ax[1], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
world.plot(ax=ax[2], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map


# for i in range(3):
#     for j in range(3):
#         ax[0].text(lon_0[i], lat_0[j], str(round(ozone_0[j][i], 4)))
#         ax[1].text(lon_1[i], lat_1[j], str(round(ozone_1[j][i], 4)))
#====================================================================
print("================================")
print(dict_['dates_for_tropospheric_column'])
print(''.join(dict_['dates_for_tropospheric_column']))
ax[0].set_title('Dates: '+ dates_0, fontsize=6)
ax[1].set_title('Dates: '+ dates_1, fontsize=6)
ax[2].set_title('Dates: '+ dates_3, fontsize=6)
#====================================================================
#====================================================================
#====================================================================
#====================================================================
#====================================================================
#====================================================================
#====================================================================
#====================================================================
#====================================================================
#====================================================================
fig2, ax2 = plt.subplots(2, 1, figsize=(12, 4))
#fig2.subplots_adjust(hspace=0, wspace=-1)

folder_0 = "/Users/joshuamiller/Documents/Lancaster/Data/Ozone-L2_O3_TCL"
f = "/Users/joshuamiller/Documents/Lancaster/Data/L2_O3_TCL/S5P_RPRO_L2__O3_TCL_20200607T121450_20200613T130036_13778_03_020401_20230329T123613/S5P_RPRO_L2__O3_TCL_20200607T121450_20200613T130036_13778_03_020401_20230329T123613.nc"

dict_ = Extract_netCDF4(f,
                        ['latitude_ccd', 'longitude_ccd', 'ozone_tropospheric_vertical_column', 'dates_for_tropospheric_column', 'qa_value'],
                        groups='all',
                        print_sum=False)


lat   = dict_['latitude_ccd']
lon   = dict_['longitude_ccd']
ozone = np.ma.getdata(np.squeeze(dict_['ozone_tropospheric_vertical_column']))
qa    = np.ma.getdata(np.squeeze(dict_['qa_value']))
dates = ''.join(dict_['dates_for_tropospheric_column'])

print(np.shape(lat), np.shape(lon), np.shape(ozone), np.shape(qa), np.argwhere(np.isnan(qa)))

lat_tiled = np.tile(lat, (np.shape(lon)[0], 1)).T
lon_tiled = np.tile(lon, (np.shape(lat)[0], 1))

upper = 100
lower = 0
new_lat = []
new_lon = []
new_ozone = []
for i in range(np.shape(ozone)[0]):
    for j in range(np.shape(ozone)[1]):
        if ((lower < ozone[i][j]) and (ozone[i][j] < upper)):
            new_lat.append(lat_tiled[i][j])
            new_lon.append(lon_tiled[i][j])
            new_ozone.append(ozone[i][j])
#====================================================================

ozone_norm = Normalize(vmin=0, vmax=1.1 * max(new_ozone))
ozone_cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'cornflowerblue', 'powderblue', 'pink', 'palevioletred', 'red'], N=200) 
qa_norm    = Normalize(vmin=min(100*qa.ravel()), vmax=max(100*qa.ravel()))
qa_cmap    = LinearSegmentedColormap.from_list('custom', ['red', 'yellow', 'green'], N=800) #8 Higher N=more smooth

new_ozone_plot = ax2[0].scatter(x=new_lon, y=new_lat, c=new_ozone, 
                                s=2, marker='o', norm=ozone_norm, cmap=ozone_cmap)
qa_plot = ax2[1].scatter(x=lon_tiled, y=lat_tiled, c=100*qa, 
                         s=2, marker='o', norm=qa_norm, cmap=qa_cmap)

world.plot(ax=ax2[0], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
world.plot(ax=ax2[1], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map

divider = make_axes_locatable(ax2[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
O3_bar = plt.colorbar(new_ozone_plot, cax=cax)
O3_bar.set_label(r'$O_3$'+' '+r'$\left( \frac{mol}{m^2} \right)$', fontsize=15, rotation=270, labelpad=30)

divider = make_axes_locatable(ax2[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
qa_bar = plt.colorbar(qa_plot, cax=cax)
qa_bar.set_label(r'$\mathbf{Quality\ Flag}$'+r' '+r'$\mathbf{(\%)}$', fontsize=10, rotation=270, labelpad=30)

ax2[0].set_ylim(-25, 25)
ax2[1].set_ylim(-25, 25)

ax2[0].xaxis.set_major_formatter(FuncFormatter(DegreeFormatter))
ax2[0].yaxis.set_major_formatter(FuncFormatter(DegreeFormatter))

ax2[1].xaxis.set_major_formatter(FuncFormatter(DegreeFormatter))
ax2[1].yaxis.set_major_formatter(FuncFormatter(DegreeFormatter))

#fig.savefig(os.path.join('/Users/joshuamiller/Documents/Lancaster/Figs', "L2__O3_TCL_interp.pdf"), bbox_inches = 'tight', pad_inches = 0)
# mask = np.ma.getmaskarray(dict_['ozone_tropospheric_vertical_column'])
# for i in range(np.shape(mask)[0]):
#     for j in range(np.shape(mask)[1]):
#         print("i =", i, ", j =", j, ", mask =", mask[i][j], ", val =", dict_['ozone_tropospheric_vertical_column'][i][j], type(dict_['ozone_tropospheric_vertical_column'][i][j]))

#2fig2.savefig(os.path.join('/Users/joshuamiller/Documents/Lancaster/Dissertation/Figs', dates+"_O3_qa.pdf"), bbox_inches='tight', pad_inches=0)
#====================================================================
#====================================================================
#====================================================================
#====================================================================
#====================================================================
#====================================================================
#====================================================================
#====================================================================
#====================================================================
#====================================================================
# ''' Create subplot array '''
# fig3 = plt.figure(figsize=(12, 4), dpi=100, constrained_layout=False)
# gs  = fig3.add_gridspec(2, 3)
# ax0 = fig3.add_subplot(gs[0, :])
# ax1 = fig3.add_subplot(gs[1, 0])
# ax2 = fig3.add_subplot(gs[1, 1])
# ax3 = fig3.add_subplot(gs[1, 2])
# fig3.subplots_adjust(wspace=.5, hspace=0)

# f_orig  = "/Users/joshuamiller/Documents/Lancaster/Data/L2_O3_TCL/S5P_RPRO_L2__O3_TCL_20200607T121450_20200613T130036_13778_03_020401_20230329T123613/S5P_RPRO_L2__O3_TCL_20200607T121450_20200613T130036_13778_03_020401_20230329T123613.nc"
# #f_small = "/Users/joshuamiller/Documents/Lancaster/Data/Whole_Area/Ozone/S5P_RPRO_L2__O3_TCL_2018-04-30_2022-07-31_Whole_Area.nc"
# f_small = "/Users/joshuamiller/Documents/Lancaster/Data/North_Land/Ozone/S5P_RPRO_L2__O3_TCL_2018-04-30_2022-07-31_North_Land.nc"
# date_idx = 769

# dict_orig = Extract_netCDF4(f_orig,
#                             ['latitude_ccd', 'longitude_ccd', 'ozone_tropospheric_vertical_column', 'dates_for_tropospheric_column', 'qa_value'],
#                             groups='all',
#                             print_sum=False)

# lat_orig   = dict_orig['latitude_ccd']
# lon_orig   = dict_orig['longitude_ccd']
# ozone_orig = np.ma.getdata(np.squeeze(dict_orig['ozone_tropospheric_vertical_column']))
# qa_orig    = np.squeeze(dict_orig['qa_value'])
# dates_orig = ''.join(dict_orig['dates_for_tropospheric_column'])

# print(np.shape(lat_orig), np.shape(lon_orig), np.shape(ozone_orig))

# lat_orig_tiled = np.tile(lat_orig, (np.shape(lon_orig)[0], 1)).T
# lon_orig_tiled = np.tile(lon_orig, (np.shape(lat_orig)[0], 1))

# upper = 100
# lower = 0
# new_lat = []
# new_lon = []
# new_ozone = []
# for i in range(np.shape(ozone_orig)[0]):
#     for j in range(np.shape(ozone_orig)[1]):
#         if ((lower < ozone_orig[i][j]) and (ozone_orig[i][j] < upper)):
#             new_lat.append(lat_orig_tiled[i][j])
#             new_lon.append(lon_orig_tiled[i][j])
#             new_ozone.append(ozone_orig[i][j])


# dict_small = Extract_netCDF4(f_small,
#                             ['lat', 'lon', 'ozone_tropospheric_vertical_column', 'start_date', 'qa_value', 'krig_mask'],
#                             groups='all',
#                             print_sum=False)

# lat_small   = dict_small['lat']
# lon_small   = dict_small['lon']
# ozone_small = np.squeeze(np.ma.getdata(dict_small['ozone_tropospheric_vertical_column']))[date_idx, :, :]
# qa_small    = np.squeeze(dict_small['qa_value'])[date_idx, :, :]
# krig_mask   = np.squeeze(dict_small['krig_mask'])[date_idx, :, :]

# start_date  = np.squeeze(dict_small['start_date'])[date_idx]
# date  = datetime(1970, 1, 1)+timedelta(days=start_date)


# print(np.shape(lat_orig), np.shape(lon_orig), np.shape(ozone_orig), np.shape(lat_small), np.shape(lon_small), np.shape(ozone_small),
#       "date=", date, ", dates:", dates)

# lat_small_tiled = np.tile(lat_small, (np.shape(lon_small)[0], 1)).T
# lon_small_tiled = np.tile(lon_small, (np.shape(lat_small)[0], 1))

# ozone_norm = Normalize(vmin=0, vmax=1.1 * max([max(new_ozone), max(ozone_small.ravel())]))
# ozone_cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'cornflowerblue', 'powderblue', 'pink', 'palevioletred', 'red'], N=200) 
# qa_norm    = Normalize(vmin=0, vmax=100)
# qa_cmap    = LinearSegmentedColormap.from_list('custom', ['red', 'yellow', 'green'], N=800) #8 Higher N=more smooth
# krig_norm  = Normalize(vmin=0, vmax=1)
# krig_cmap  = LinearSegmentedColormap.from_list('custom', ['white', 'red'], N=2) #8 Higher N=more smooth

# new_ozone_plot = ax0.scatter(x=new_lon, y=new_lat, c=new_ozone, 
#                              s=2, marker='o', norm=ozone_norm, cmap=ozone_cmap)
# divider = make_axes_locatable(ax0)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(new_ozone_plot, cax=cax, label=r'$O_3$'+" conc. "+r'$\left(\frac{mol}{m^2}\right)$')
# PlotBoxes("data_utils/data_utils_config.yml", ax0)

# small_ozone_plot = ax1.scatter(x=lon_small_tiled, y=lat_small_tiled, c=ozone_small,
#                                s=2, marker='o', norm=ozone_norm, cmap=ozone_cmap)
# divider = make_axes_locatable(ax1)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(small_ozone_plot, cax=cax, label=r'$O_3$'+" conc. "+r'$\left(\frac{mol}{m^2}\right)$')

# qa_plot = ax2.scatter(x=lon_small_tiled, y=lat_small_tiled, c=100*qa_small, 
#                       s=2, marker='o', norm=qa_norm, cmap=qa_cmap)
# divider = make_axes_locatable(ax2)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(qa_plot, cax=cax, label='Quality flag (%)')

# krig_plot = ax3.scatter(x=lon_small_tiled, y=lat_small_tiled, c=krig_mask, 
#                         s=2, marker='o', norm=krig_norm, cmap=krig_cmap)
# divider = make_axes_locatable(ax3)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(krig_plot, cax=cax, label='Krig mask')

# world.plot(ax=ax0, facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
# world.plot(ax=ax1, facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
# world.plot(ax=ax2, facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
# world.plot(ax=ax3, facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map

# ax0.set_title(dates_orig)
# ax1.set_title(date)
# ax2.set_title(date)
# ax3.set_title(date)

# ax0.set_ylim(-25, 25)
# ax1.set_xlim(-21, 61)
# ax1.set_ylim(-21, 21)
# ax2.set_xlim(-21, 61)
# ax2.set_ylim(-21, 21)
# ax3.set_xlim(-21, 61)
# ax3.set_ylim(-21, 21)
# #====================================================================
# #====================================================================
# #====================================================================
# #====================================================================
# #====================================================================
# #====================================================================
# #====================================================================
# #====================================================================
# #====================================================================
# #====================================================================
# fig4, ax4 = plt.subplots(5, 1, figsize=(12, 7))
# #f_new = "/Users/joshuamiller/Documents/Lancaster/Data/Filtered_L2_O3_TCL_v2/S5P_RPRO_L2__O3_TCL_2020-06-07_2020-06-13_filt.nc"
# f_new = "/Users/joshuamiller/Documents/Lancaster/Data/Kriged_L2_O3_TCL_v2/S5P_RPRO_L2__O3_TCL_2020-06-07_2020-06-13_kriged_600.nc"
# dict_new = Extract_netCDF4(f_new,
#                             ['lat', 'lon', 'ozone_tropospheric_vertical_column', 'dates_for_tropospheric_column',
#                              'qa_value',
#                              'krig_mask'],
#                             groups='all')

# lat_new   = np.squeeze(dict_new['lat'])
# lon_new   = np.squeeze(dict_new['lon'])
# ozone_new = np.squeeze(dict_new['ozone_tropospheric_vertical_column'])
# qa_new    = np.squeeze(dict_new['qa_value'])
# krig_mask = np.squeeze(dict_new['krig_mask'])

# lat_new_tiled = np.tile(lat_new, (np.shape(lon_new)[0], 1)).T
# lon_new_tiled = np.tile(lon_new, (np.shape(lat_new)[0], 1))

# new_ozone_plot = ax4[0].scatter(x=new_lon, y=new_lat, c=new_ozone, 
#                                 s=2, marker='o', norm=ozone_norm, cmap=ozone_cmap)
# qa_orig_plot = ax4[1].scatter(x=lon_tiled, y=lat_tiled, c=100*qa, 
#                               s=2, marker='o', norm=qa_norm, cmap=qa_cmap)
# ozone_new_plot = ax4[2].scatter(x=lon_new_tiled, y=lat_new_tiled, c=ozone_new,
#                                  s=2, marker='o', norm=ozone_norm, cmap=ozone_cmap)
# qa_new_plot = ax4[3].scatter(x=lon_new_tiled, y=lat_new_tiled, c=100*qa_new, 
#                               s=2, marker='o', norm=qa_norm, cmap=qa_cmap)
# krig_plot = ax4[4].scatter(x=lon_new_tiled, y=lat_new_tiled, c=krig_mask, 
#                            s=2, marker='o', norm=krig_norm, cmap=krig_cmap)

# divider = make_axes_locatable(ax4[0])
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(new_ozone_plot, cax=cax, label=r'$O_3$'+" conc. "+r'$\left(\frac{mol}{m^2}\right)$')

# divider = make_axes_locatable(ax4[1])
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(qa_orig_plot, cax=cax, label='Quality flag (%)')

# divider = make_axes_locatable(ax4[2])
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(ozone_new_plot, cax=cax, label=r'$O_3$'+" conc. "+r'$\left(\frac{mol}{m^2}\right)$')

# divider = make_axes_locatable(ax4[3])
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(qa_new_plot, cax=cax, label='Quality flag (%)')

# divider = make_axes_locatable(ax4[4])
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(krig_plot, cax=cax, label='krig mask')

# world.plot(ax=ax4[0], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
# world.plot(ax=ax4[1], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
# world.plot(ax=ax4[2], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
# world.plot(ax=ax4[3], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
# world.plot(ax=ax4[4], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map

# ax4[0].set_ylim(-25, 25)
# ax4[1].set_ylim(-25, 25)
# ax4[2].set_ylim(-25, 25)
# ax4[3].set_ylim(-25, 25)
# ax4[4].set_ylim(-25, 25)

plt.show()