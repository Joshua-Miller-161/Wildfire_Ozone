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
import yaml
    
sys.path.append(os.getcwd())
from data_utils.extract_netCDF4 import Extract_netCDF4
from misc.misc_utils import GetBoxCoords
from vis.plotting_utils import PlotBoxes
#====================================================================
save_folder = '/Users/joshuamiller/Documents/Lancaster/Data/'
data_folder = '/Users/joshuamiller/Documents/Lancaster/Data/Kriged_L2_O3_TCL'
#====================================================================
boxes = GetBoxCoords("data_utils/data_utils_config.yml")
#====================================================================
''' Filter into the boxes '''
# for file in os.listdir(data_folder):
#     if file.endswith('.nc'):
#         #------------------------------------------------------------
#         dict_ = Extract_netCDF4(os.path.join(data_folder, file),
#                                 ['lat', 'lon', 'ozone_tropospheric_vertical_column', 'dates_for_tropospheric_column'],
#                                 groups='all',
#                                 print_sum=False)
#         #print(">> Opened:", os.path.join(data_folder, file))

#         lat = dict_['lat']
#         lon = dict_['lon']
#         ozone = np.squeeze(dict_['ozone_tropospheric_vertical_column'])
#         dates = ''.join(dict_['dates_for_tropospheric_column'])
#         dates = dates.split(' ')
#         lat_tiled = np.tile(lat, (np.shape(lon)[0], 1)).T
#         lon_tiled = np.tile(lon, (np.shape(lat)[0], 1))
#         #------------------------------------------------------------
#         for box_name in boxes.keys():
#             #print(" >> Searching in", box_name)
#             new_lat = []
#             new_lon = []
#             new_O3  = []
#             for i in range(np.shape(ozone)[0]):
#                 for j in range(np.shape(ozone)[1]):
#                     if (boxes[box_name][3] <= lat_tiled[i,j] and lat_tiled[i,j] <= boxes[box_name][1] and  boxes[box_name][0] <= lon_tiled[i, j] and lon_tiled[i, j] <= boxes[box_name][2]):
#                         new_lat.append(lat_tiled[i,j])
#                         new_lon.append(lon_tiled[i,j])
#                         new_O3.append(ozone[i,j])

        
#             df = pd.DataFrame(np.array([new_lat, new_lon, new_O3]).T, 
#                               columns=['lat', 'lon', 'ozone_tropospheric_vertical_column'])
            
#             path = save_folder+'/'+box_name+'/'+'L2__O3_TCL'
#             filename = file[:37]+'_'+box_name+'.csv'
#             #print(filename)
#             df.to_csv(os.path.join(path, filename), index=False)
            #print("  >> Saved:", os.path.join(save_folder, box_name), ".csv |", str(len(new_O3)), "values in box")
#====================================================================
fig, ax = plt.subplots(2, 1, figsize=(8, 6))

ozone_norm = Normalize(vmin=0, vmax=0.024)
ozone_cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'deepskyblue', 'lavenderblush', 'pink', 'red'], N=200) #8 Higher N=more smooth

dict_ = Extract_netCDF4('/Users/joshuamiller/Documents/Lancaster/Data/Kriged_L2_O3_TCL/S5P_RPRO_L2__O3_TCL_20200517-20200521_kriged_382.nc',
                        ['lat', 'lon', 'ozone_tropospheric_vertical_column', 'dates_for_tropospheric_column'],
                        groups='all',
                        print_sum=True)

lat_3 = dict_['lat']
lon_3 = dict_['lon']
ozone_3 = np.squeeze(dict_['ozone_tropospheric_vertical_column'])
dates_3 = ''.join(dict_['dates_for_tropospheric_column'])
dates_3 = dates_3.split(' ')
lat_tiled_3 = np.tile(lat_3, (np.shape(lon_3)[0], 1)).T
lon_tiled_3 = np.tile(lon_3, (np.shape(lat_3)[0], 1))
O3_3 = ax[0].scatter(x=lon_tiled_3, y=lat_tiled_3, c=ozone_3,
                        s=1, marker='.', norm=ozone_norm, cmap=ozone_cmap)

divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(O3_3, cax=cax, label=r'$O_3$'+' concentration '+r'$\left(\frac{mol}{m^2}\right)$')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=90)
ax[0].set_title("Ozone: " + dates_3[0][:4] + '-' + dates_3[0][4:6] + '-' + dates_3[0][6:8] + " - " + dates_3[-1][:4] + '-' + dates_3[-1][4:6] + '-' + dates_3[-1][6:8])



df = pd.read_csv('/Users/joshuamiller/Documents/Lancaster/Data/South_Land/L2__O3_TCL/S5P_RPRO_L2__O3_TCL_20200517-20200521_South_Land.csv')
O3_small = ax[1].scatter(x=df['lon'], y=df['lat'], c=df['ozone_tropospheric_vertical_column'],
                         s=1, marker='.', norm=ozone_norm, cmap=ozone_cmap)



world = gpd.read_file("/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")
world.plot(ax=ax[0], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
world.plot(ax=ax[1], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
PlotBoxes("data_utils/data_utils_config.yml", ax[0])
PlotBoxes("data_utils/data_utils_config.yml", ax[1])
ax[0].set_xlim(-30, 70)
ax[0].set_ylim(-30, 30)
ax[1].set_xlim(-30, 70)
ax[1].set_ylim(-30, 30)

plt.show()