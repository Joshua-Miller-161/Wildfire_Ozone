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
import os
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
                        print_keys=True,
                        print_sum=True,
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
lon = np.tile(dict_['longitude'], (np.shape(dict_['latitude'])[0], 1))

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
            print("min geo.:", min_geo, ", max geo.:", max_geo)
            geo_norm = Normalize(vmin=min_geo, vmax=max_geo)
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            geo_plot = ax[i][j].scatter(x=lon.ravel()-360, y=lat.ravel(), c=geo_pot, 
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
            print("min alt.:", altitude, ", max alt.:", altitude)
            alt_norm = Normalize(vmin=min_alt, vmax=max_alt)
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            alt_plot = ax[i][j].scatter(x=lon.ravel()-360, y=lat.ravel(), c=altitude, 
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

plt.show()

fig.savefig(os.path.join('/Users/joshuamiller/Documents/Lancaster/Figs', "GeoPot_Alt.pdf"), bbox_inches = 'tight', pad_inches = 0)