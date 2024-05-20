import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import numpy as np
import geopandas as gpd
from matplotlib.colors import Normalize, LogNorm
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Point
import numpy as np
import iris_grib
import sys
from cartopy import crs

sys.path.append(os.getcwd())
from data_utils.extraction_funcs import ExtractGRIBIris
#====================================================================
levels = [1, 25, 50, 100, 137]
#====================================================================
''' Get data '''
path = "/Users/joshuamiller/Documents/Python Files/Wildfire_Ozone/data_utils/era5_utils/z_on_ml.grib"


cubes = iris_grib.load_cubes(path)
cubes = list(cubes)
print("============")
print(cubes[0], np.shape(cubes[0].data))
print("============")

dict_ = ExtractGRIBIris(path,
                        var_names='all',
                        sclice_over_var='model_level_number',
                        print_keys=False,
                        print_sum=False,
                        num_examples=2,
                        use_dask_array=False)
print("-_- -_- -_- -_- -_- -_- -_- -_- -_- -_-")
print(np.shape(dict_['geopotential']), np.shape(dict_['longitude']))
#====================================================================
''' Set up plot '''
fig, ax = plt.subplots(len(levels), 2, figsize=(8, 8))
fig.subplots_adjust(wspace=.5, hspace=.9)
#====================================================================
''' Geopotential '''
lat = np.tile(dict_['latitude'], (np.shape(dict_['longitude'])[0], 1)).T
lon = np.tile(dict_['longitude']-360, (np.shape(dict_['latitude'])[0], 1))

points = [Point(x,y) for x,y in zip(lon.ravel(), lat.ravel())]
print(" + = + = +", np.shape(lat), np.shape(lon), np.shape(points), np.shape(dict_['geopotential'][0, :, :]))
#====================================================================
''' Get world map '''
world = gpd.read_file("/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")
#====================================================================
''' Plot '''
for i in range(len(levels)):
    for j in range(2):
        #------------------------------------------------------------
        if (j == 0):
            ''' Geopotential '''
            geo_pot = dict_['geopotential'][137-levels[i], :, :].ravel()
            min_geo = min(geo_pot)
            max_geo = max(geo_pot)
            #print("min geo.:", min_geo, ", max geo.:", max_geo)
            geo_norm = Normalize(vmin=min_geo, vmax=max_geo)
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            geo_plot = ax[i][j].scatter(x=lon.ravel(), y=lat.ravel(), c=geo_pot, 
                                        norm=geo_norm, cmap=mpl.colormaps['spring'], marker='.', s=20)
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            divider = make_axes_locatable(ax[i][j])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(geo_plot, cax=cax)
            ax[i][j].set_title("Level: "+str(levels[i])+" Geopot.: avg. "+str(round(np.mean(geo_pot), 3))+' '+r'$\frac{m^2}{s^2}$')
        
        else:
            ''' Altitude '''
            geo_pot  = dict_['geopotential'][137-levels[i], :, :].ravel()
            altitude = (geo_pot * 6371000) / (9.8 * 6371000 - geo_pot)
            min_alt = min(altitude)
            max_alt = max(altitude)
            #print("min alt.:", altitude, ", max alt.:", altitude)
            alt_norm = Normalize(vmin=min_alt, vmax=max_alt)
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            alt_plot = ax[i][j].scatter(x=lon.ravel(), y=lat.ravel(), c=altitude, 
                                        norm=alt_norm, cmap=mpl.colormaps['winter'], marker='.', s=20)
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            divider = make_axes_locatable(ax[i][j])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(alt_plot, cax=cax)
            ax[i][j].set_title("Level: "+str(levels[i])+" Alt.: avg. "+str(round(np.mean(altitude), 3))+' '+r'$m$')
        #------------------------------------------------------------
        ''' Format '''
        ax[i][j].set_xlim(-20, 60)
        ax[i][j].set_ylim(-20, 20)
        #------------------------------------------------------------
        world.plot(ax=ax[i][j], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True) # GOOD lots the map
#fig.savefig(os.path.join('/Users/joshuamiller/Documents/Lancaster/Figs', "GeoPot_Alt.pdf"), bbox_inches = 'tight', pad_inches = 0)
#====================================================================      
fig2, ax2 = plt.subplots(4, 6, figsize=(14, 7))
fig2.subplots_adjust(wspace=.2, hspace=0)
levels = [137, 136, 135, 134, 
          133, 132, 131, 130, 
          129, 128, 127, 126,
          125, 124, 123, 122,
          121, 120, 119, 118,
          117, 116, 115, 114]

locs = [[-15, 0],
        [20, 0]]
point_alt = np.ones((len(levels), len(locs)), float) * -999

geo_pot  = dict_['geopotential'][137-max(levels):137-min(levels), :, :]
altitude = (geo_pot * 6371000) / (9.8 * 6371000 - geo_pot)
max_alt = max(altitude.ravel())
#print("min alt.:", altitude, ", max alt.:", altitude)
alt_norm = Normalize(vmin=0, vmax=5000)
alt_cmap = LinearSegmentedColormap.from_list('custom', ['indigo', 'blue', 'yellow', 'orange', 'red'], N=100) # Higher N=more smooth

for i in range(4):
    for j in range(6):
        geo_pot  = dict_['geopotential'][137-levels[6*i+j], :, :]
        altitude = (geo_pot * 6371000) / (9.8 * 6371000 - geo_pot)
        #------------------------------------------------------------
        alt = ax2[i][j].scatter(x=lon, y=lat, c=altitude, 
                                norm=alt_norm, cmap=alt_cmap, marker='.', s=20)
        
        for k in range(len(locs)):
            ax2[i][j].scatter(x=locs[k][0], y=locs[k][1], color='black', marker='+')
            lon_idx = np.where(lon==locs[k][0])[1][0]
            lat_idx = np.where(lat==locs[k][1])[0][1]
            #print("$$$$$", np.where(lon==locs[k][0])[1][0],  np.where(lat==locs[k][1])[0][1])
            point_alt[6*i+j][k] = altitude[lat_idx][lon_idx]

        world.plot(ax=ax2[i][j], facecolor='none', edgecolor='black', linewidth=.5, alpha=1, legend=True)
        #------------------------------------------------------------
        ''' Format '''
        ax2[i][j].set_xlim(-21, 61)
        ax2[i][j].set_ylim(-21, 21)
        ax2[i][j].set_title(str(levels[6*i+j])+" Alt.: avg. "+str(round(np.mean(altitude))))

# Create an axis on the left side for the colorbar
fig2.subplots_adjust(left=0.1, right=0.8)
cbar_ax = fig2.add_axes([0.81, 0.14, 0.02, 0.71])
cbar = fig2.colorbar(alt, cax=cbar_ax)
cbar.set_label('Altitude (m)', rotation=270, labelpad=15)
#==================================================================== 
fig3, ax3 = plt.subplots(len(locs), 1, figsize=(14, 7))
lol = 2
for i in range(len(locs)):
    ax3[i].plot(levels, point_alt[:, i], label=str(locs[i]))
    ax3[i].plot(levels, point_alt[lol, i]*np.ones_like(levels), color='black')
    ax3[i].legend(loc='upper right')
    ax3[i].text(min(levels)+1, min(point_alt[:, i])+10, str(round(point_alt[lol, i] - min(point_alt[:, i]), 3)))
plt.show()