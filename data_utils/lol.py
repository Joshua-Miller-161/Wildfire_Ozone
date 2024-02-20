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
    
sys.path.append(os.getcwd())
from data_utils.extraction_funcs import Extract_netCDF4
#====================================================================
''' World map '''
world = gpd.read_file("/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")
crs = world.crs
print(" + + +", crs, crs.datum)
#====================================================================
# f_fire = ['/Users/joshuamiller/Documents/Lancaster/Data/Kriged_MODIS_C61/MODIS_C61_2018-07-01_kriged.csv',
#           '/Users/joshuamiller/Documents/Lancaster/Data/Kriged_MODIS_C61/MODIS_C61_2018-07-02_kriged.csv',
#           '/Users/joshuamiller/Documents/Lancaster/Data/Kriged_MODIS_C61/MODIS_C61_2020-11-17_kriged.csv',
#           '/Users/joshuamiller/Documents/Lancaster/Data/Kriged_MODIS_C61/MODIS_C61_2020-11-18_kriged.csv']
# f_O3 = ['/Users/joshuamiller/Documents/Lancaster/Data/Kriged_L2_O3_TCL/S5P_RPRO_L2__O3_TCL_20180701-20180705_kriged_555.nc',
#         '/Users/joshuamiller/Documents/Lancaster/Data/Kriged_L2_O3_TCL/S5P_RPRO_L2__O3_TCL_20180702-20180705_kriged_527.nc',
#         '/Users/joshuamiller/Documents/Lancaster/Data/Kriged_L2_O3_TCL/S5P_RPRO_L2__O3_TCL_20201117-20201120_kriged_497.nc',
#         '/Users/joshuamiller/Documents/Lancaster/Data/Kriged_L2_O3_TCL/S5P_RPRO_L2__O3_TCL_20201118-20201121_kriged_445.nc']



f_fire = ['/Users/joshuamiller/Documents/Lancaster/Data/Kriged_MODIS_C61/MODIS_C61_2020-05-17_kriged.csv',
          '/Users/joshuamiller/Documents/Lancaster/Data/Kriged_MODIS_C61/MODIS_C61_2020-11-17_kriged.csv']
f_O3 = ['/Users/joshuamiller/Documents/Lancaster/Data/Kriged_L2_O3_TCL/S5P_RPRO_L2__O3_TCL_20200517-20200521_kriged_382.nc',
        '/Users/joshuamiller/Documents/Lancaster/Data/Kriged_L2_O3_TCL/S5P_RPRO_L2__O3_TCL_20201117-20201120_kriged_497.nc']






f_O3_ = ['/Users/joshuamiller/Documents/Lancaster/Data/Kriged_L2_O3_TCL/S5P_RPRO_L2__O3_TCL_20180527-20180530_kriged_243.nc',
         '/Users/joshuamiller/Documents/Lancaster/Data/Kriged_L2_O3_TCL/S5P_RPRO_L2__O3_TCL_20180526-20180529_kriged_199.nc',
         '/Users/joshuamiller/Documents/Lancaster/Data/Kriged_L2_O3_TCL/S5P_RPRO_L2__O3_TCL_20180524-20180528_kriged_322.nc']

for file in f_O3_:
    dict_ = Extract_netCDF4(file,
                        ['lat', 'lon', 'ozone_tropospheric_vertical_column', 'dates_for_tropospheric_column'],
                        groups='all',
                        print_sum=True)
    print(dict_['dates_for_tropospheric_column'])
    print("+++++++++++++++++++++++++++++++++++")
    print("+++++++++++++++++++++++++++++++++++")
    print("+++++++++++++++++++++++++++++++++++")
    print("+++++++++++++++++++++++++++++++++++")
#====================================================================
fig, ax = plt.subplots(len(f_fire), 2, figsize=(9,6)) # figsize (height (in), width (in))
fig.subplots_adjust(wspace=1, hspace=-.1)
#====================================================================
fire_norm = Normalize(vmin=0.1, vmax=10**3)
fire_cmap = LinearSegmentedColormap.from_list('custom', ['yellow', 'orange', 'red'], N=200) # Higher N=more smooth

ozone_norm = Normalize(vmin=0, vmax=0.024)
ozone_cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'deepskyblue', 'lavenderblush', 'pink', 'red'], N=200) #8 Higher N=more smooth
#====================================================================
for i in range(len(f_fire)):
    new_fire_df = pd.read_csv(f_fire[i])

    frp = []
    lat = []
    lon = []
    frp_0 = []
    lat_0 = []
    lon_0 = []
    for j in range(new_fire_df.shape[0]):
        if (new_fire_df.loc[j, 'frp'] > 0.1):
            frp.append(new_fire_df.loc[j, 'frp'])
            lat.append(new_fire_df.loc[j, 'lat'])
            lon.append(new_fire_df.loc[j, 'lon'])
        else:
            frp_0.append(new_fire_df.loc[j, 'frp'])
            lat_0.append(new_fire_df.loc[j, 'lat'])
            lon_0.append(new_fire_df.loc[j, 'lon'])

    new_fire = ax[i][0].scatter(x=lon, y=lat, c=frp, 
                                s=0.5, marker='.', cmap=fire_cmap, norm=fire_norm)

    divider = make_axes_locatable(ax[i][0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(new_fire, cax=cax, label='MaxFRP (Megawatts)')
    cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=180)

    ax[i][0].scatter(x=lon_0, y=lat_0, s=0.5, marker='.', color='black')

    ax[i][0].set_title("Fire: "+f_fire[i][-21:-11])
    #================================================================
    dict_ = Extract_netCDF4(f_O3[i],
                            ['lat', 'lon', 'ozone_tropospheric_vertical_column', 'dates_for_tropospheric_column'],
                            groups='all',
                            print_sum=False)

    lat_3 = dict_['lat']
    lon_3 = dict_['lon']
    ozone_3 = np.squeeze(dict_['ozone_tropospheric_vertical_column'])
    dates_3 = ''.join(dict_['dates_for_tropospheric_column'])
    dates_3 = dates_3.split(' ')
    lat_tiled_3 = np.tile(lat_3, (np.shape(lon_3)[0], 1)).T
    lon_tiled_3 = np.tile(lon_3, (np.shape(lat_3)[0], 1))
    O3_3 = ax[i][1].scatter(x=lon_tiled_3, y=lat_tiled_3, c=ozone_3,
                            s=0.5, marker='.', norm=ozone_norm, cmap=ozone_cmap)

    divider = make_axes_locatable(ax[i][1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(O3_3, cax=cax, label=r'$O_3$'+' concentration '+r'$\left(\frac{mol}{m^2}\right)$')
    cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=90)
    ax[i][1].set_title("Ozone: " + dates_3[0][:4] + '-' + dates_3[0][4:6] + '-' + dates_3[0][6:8] + " - " + dates_3[-1][:4] + '-' + dates_3[-1][4:6] + '-' + dates_3[-1][6:8])

for i in range(len(f_fire)):
    for j in range(2):
        world.plot(ax=ax[i][j], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
        ax[i][j].set_xlim(-20, 60)
        ax[i][j].set_ylim(-20, 20)
        
plt.show()

fig.savefig('/Users/joshuamiller/Documents/Lancaster/Figs/Fire_Ozone_Camparison.pdf', bbox_inches = 'tight', pad_inches = 0)