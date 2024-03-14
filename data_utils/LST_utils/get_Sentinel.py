import os
os.environ['USE_PYGEOS'] = '0'
import pystac_client
import planetary_computer
import geopandas as gpd
import sys

sys.path.append(os.getcwd())
from misc.misc_utils import GetBoxCoords
#====================================================================
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

box = GetBoxCoords("/Users/joshuamiller/Documents/Python Files/Wildfire_Ozone/data_utils/data_utils_config.yml")
box = box['Whole_Area']
min_lon = box[0]
max_lat = box[1]
max_lon = box[2]
min_lat = box[3]

area_of_interest = {
    "type": "Polygon",
    "coordinates": [[[min_lon, min_lat],
                     [min_lon, max_lat],
                     [max_lon, max_lat],
                     [max_lon, min_lon],
                     [min_lon, min_lon]]],
}

time_range = "2020-12-01"

search = catalog.search(
    collections=["sentinel-3-slstr-lst-l2-netcdf"], intersects=area_of_interest, datetime=time_range
)
items = search.item_collection()
print(len(items))

