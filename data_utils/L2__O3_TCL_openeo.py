# Import openeo and connect to Sentinel Hub
import openeo
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
from shapely.geometry import Point
import netCDF4 as nc
import numpy as np
import sys
import os

sys.path.append(os.getcwd())
from data_utils.extraction_funcs import Extract_netCDF4
#====================================================================
connection = openeo.connect("openeo.dataspace.copernicus.eu")
connection.authenticate_oidc()

print(connection.list_collection_ids())
collection = "SENTINEL_5P_L2"
print("---")

print(connection.describe_collection("SENTINEL_5P_L2"))

# # Specify the date and collection
#date = "2020-01-01"

# # Load the data as a datacube object
# datacube = connection.load_collection(
#      collection_id=collection,
#      spatial_extent={"west": -180, "east": 180, "south": -90, "north": 90},
#      temporal_extent=[date, date])

#O3_datacube = datacube.band("O3")

O3_datacube = connection.load_collection(collection_id=collection, bands='O3')
O3_datacube = O3_datacube.filter_bbox(west=-20, south=-20, east=60, north=20)
O3_datacube = O3_datacube.filter_temporal(extent=["2020-01-28T11:18:27Z", "2020-02-03T12:04:40Z"])

# # Select the ozone column band
print("============================")
print("============================")
print(connection.list_file_formats())
#O3_datacube.download("ozone_20200128.nc", format="netCDF")
result = O3_datacube.save_result("netCDF")


job = result.create_job()
job.start_and_wait()
job.get_results().download_files("output")
#====================================================================
#====================================================================
''' Get data '''
dict_ = Extract_netCDF4("output/openEO.nc",
                        ['y', 'x', 't', 'O3', 'crs'],
                        groups='all',
                        print_sum=True)

print("============================================================")
print(dict_)
#====================================================================
''' Make subplot '''
fig, ax = plt.subplots(figsize=(8, 6))

#ax.set_xlim(min(dict_['longitude_ccd']) - .1, max(dict_['longitude_ccd']) + .1)
#ax.set_ylim(min(dict_['latitude_ccd']) - .1, max(dict_['latitude_ccd']) + .1)
#====================================================================
''' Plot world map '''
world = gpd.read_file("/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")

world.plot(ax=ax, color='white', edgecolor='black', linewidth=0.1, alpha=1, legend=True) # GOOD lots the map
#====================================================================
''' Plot ozone '''
date = 0
ozone = dict_['O3'][date, :, :]

# - - - - - - - - - Get points for the ozone plot - - - - - - - - - - -
lat = np.tile(dict_['y'], (np.shape(dict_['x'])[0], 1)).T
lon = np.tile(dict_['x'], (np.shape(dict_['y'])[0], 1))

points = [Point(x,y) for x,y in zip(lon.ravel(), lat.ravel())]
points_gdf = gpd.GeoDataFrame(geometry=points)

print('lat:', np.shape(lat),
      ', lon:', np.shape(lon),
      ', ozone:', np.shape(ozone),
      ', points:', np.shape(points))

del(lat)
del(lon)

ozone_gdf = gpd.GeoDataFrame(geometry=points).assign(data=ozone.ravel())

# - - - - - - - - - - - Make colorbar for ozone - - - - - - - - - - -
ozone_norm = Normalize(vmin=0, vmax=max(dict_['O3'].ravel()))
ozone_cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'red'], N=200) # Higher N=more smooth

# - - - - - - - - - - - - - Plot ozone data - - - - - - - - - - - - -
ozone_gdf.plot(ax=ax, column='data', cmap=ozone_cmap, norm=ozone_norm, markersize=5, alpha=1, legend=True)
#====================================================================
plt.title(str(dict_['t']))

#====================================================================
plt.show()


#====================================================================


#====================================================================
