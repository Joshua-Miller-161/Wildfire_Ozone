import cdsapi
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.getcwd())
from misc.misc_utils import GetBoxCoords
#====================================================================
def GetDatesInRange(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    date_generated = [start + timedelta(days=x) for x in range(0, (end-start).days + 1)]

    return [date.strftime("%Y-%m-%d") for date in date_generated]
#====================================================================
def DownloadERA5(dest_folder, params, levels, dates,
                 min_lat, max_lat, min_lon, max_lon, short_names):
    assert len(params) == len(short_names), "len(params) must equal len(short_names). Got "+str(len(params)+' and '+str(len(short_names)))
    #----------------------------------------------------------------
    for i in range(len(params)):
        params[i] = str(params[i])

    params_str   = ''
    params_str_f = ''
    for param in params:
        params_str += str(int(param)) + '/'
        params_str_f += str(int(param)) + '-'
    
    params_str   = params_str[:-1]
    params_str_f = params_str_f[:-1]
    #----------------------------------------------------------------
    short_names_str_f = ''
    for short_name in short_names:
        short_names_str_f += short_name + '-'
    
    short_names_str_f = short_names_str_f[:-1]
    #----------------------------------------------------------------
    levels_str   = ''
    levels_str_f = ''
    for level in levels:
        levels_str += str(int(level)) +'/'
        levels_str_f += str(int(level)) + '-'

    levels_str   = levels_str[:-1] # Get rid of last /
    levels_str_f = levels_str_f[:-1]
    #----------------------------------------------------------------
    bbox = [max_lat, min_lon, min_lat, max_lon]
    #----------------------------------------------------------------
    time = '14:00:00'

    print('params_str=', params_str, ', levels_str=', levels_str, ', short_name=', short_names_str_f)
    
    download_dict = {               # Requests follow MARS syntax
                                    # Keywords 'expver' and 'class' can be dropped. They are obsolete
                                    # since their values are imposed by 'reanalysis-era5-complete'
        'date'    : '',             # The hyphens can be omitted
        'levelist': levels_str,     # 1 is top level, 137 the lowest model level in ERA5. Use '/' to separate values.
        'levtype' : 'ml',
        'param'   : params_str,     # Full information at https://apps.ecmwf.int/codes/grib/param-db/
                                    # The native representation for temperature is spherical harmonics
        'stream'  : 'oper',         # Denotes ERA5. Ensemble members are selected by 'enda'
        'time'    : time,     # You can drop :00:00 and use MARS short-hand notation, instead of '00/06/12/18' | Josh's note: 10:00:00 should indicate noon in Cape Town, SA, since time is measured from GMT
        'type'    : 'an',
        'area'    : bbox,           # North, West, South, East. Default: global
        'grid'    : [1.0, 0.5],     # [longitude spacing, latitude spacing]. Default: spherical harmonics or reduced Gaussian grid
        'format'  : 'grib',         # Output needs to be regular lat-lon, so only works in combination with 'grid'!
    }
    #----------------------------------------------------------------
    for date in dates:
        assert datetime.strptime(date, "%Y-%m-%d"), date+" has invalid date format. Need YYYY-MM-DD."

        download_dict['date'] = date
        #print(download_dict)
        c = cdsapi.Client()
        print("________________________________________________________________________________")
        print(">> Downloading", 'ERA5_p='+short_names_str_f+'_l='+levels_str_f+'_'+date+'_'+time+'.grib', '\n')
        c.retrieve('reanalysis-era5-complete', 
                   download_dict,
                   os.path.join(dest_folder, 'ERA5_p='+short_names_str_f+'_l='+levels_str_f+'_'+date+'_'+time+'.grib'))
        print("\n>> Saved", 'ERA5_p='+short_names_str_f+'_l='+levels_str_f+'_'+date+'_'+time+'.grib')

#====================================================================

# 130: Temperature (K)
# 3015: Maximum temperature (K)
# 131: U-component of wind (m/s)
# 132: V-component of wind (m/s)
# min_lon, max_lat, max_lon, min_lat

var = 130
level = '137'
dest_folder = ''
short_name = ''

if (var == 130):
    dest_folder = 'Temp-ERA5'
    short_name = 'temp'
elif (var == 131):
    dest_folder = 'Uwind-ERA5'
    short_name = 'Uwind'
elif (var == 132):
    dest_folder = 'Vwind-ERA5'
    short_name = 'Vwind'

start_date = "2018-04-30"
end_date = "2022-07-31"
date_list = GetDatesInRange(start_date, end_date)
date_list_str = "/".join(date_list)
#print(date_list_str)
#====================================================================

# DownloadERA5(dest_folder=os.path.join("/Users/joshuamiller/Documents/Lancaster/Data", dest_folder),
#              dates=date_list,
#              min_lat=-19.75, min_lon=-19.5, max_lat=19.75, max_lon=59.5,
#              levels=[135],
#              params=[var],
#              short_names=[short_name])

download_dict = {               # Requests follow MARS syntax
                                # Keywords 'expver' and 'class' can be dropped. They are obsolete
                                # since their values are imposed by 'reanalysis-era5-complete'
    'date'    : date_list_str,             # The hyphens can be omitted
    'levelist': level,     # 1 is top level, 137 the lowest model level in ERA5. Use '/' to separate values.
    'levtype' : 'ml',
    'param'   : str(var),     # Full information at https://apps.ecmwf.int/codes/grib/param-db/
                                # The native representation for temperature is spherical harmonics
    'stream'  : 'oper',         # Denotes ERA5. Ensemble members are selected by 'enda'
    'time'    : '14:00:00',     # You can drop :00:00 and use MARS short-hand notation, instead of '00/06/12/18' | Josh's note: 14:00:00 should indicate noon in Cape Town, SA, since time is measured from GMT
    'type'    : 'an',
    'area'    : [19.75, -19.5, -19.75, 59.5],           # North, West, South, East. Default: global
    'grid'    : [1.0, 0.5],     # Latitude/longitude. Default: spherical harmonics or reduced Gaussian grid
    'format'  : 'grib',         # Output needs to be regular lat-lon, so only works in combination with 'grid'!
}
c = cdsapi.Client()
c.retrieve('reanalysis-era5-complete', 
            download_dict,
            os.path.join('/Users/joshuamiller/Documents/Lancaster/Data/'+dest_folder, short_name+'_l='+level+'_'+start_date+'_'+end_date+'_14:00:00'+'.grib'))


# download_dict = {               # Requests follow MARS syntax
#                                 # Keywords 'expver' and 'class' can be dropped. They are obsolete
#                                 # since their values are imposed by 'reanalysis-era5-complete'
#     'date'    : '2018-04-28',             # The hyphens can be omitted
#     'levelist': level,     # 1 is top level, 137 the lowest model level in ERA5. Use '/' to separate values.
#     'levtype' : 'ml',
#     'param'   : str(var),     # Full information at https://apps.ecmwf.int/codes/grib/param-db/
#                                 # The native representation for temperature is spherical harmonics
#     'stream'  : 'oper',         # Denotes ERA5. Ensemble members are selected by 'enda'
#     'time'    : '14:00:00',     # You can drop :00:00 and use MARS short-hand notation, instead of '00/06/12/18' | Josh's note: 14:00:00 should indicate noon in Cape Town, SA, since time is measured from GMT
#     'type'    : 'an',
#     'area'    : [19.75, -19.5, -19.75, 59.5],           # North, West, South, East. Default: global
#     'grid'    : [1.0, 0.5],     # Latitude/longitude. Default: spherical harmonics or reduced Gaussian grid
#     'format'  : 'grib',         # Output needs to be regular lat-lon, so only works in combination with 'grid'!
# }
# c = cdsapi.Client()
# c.retrieve('reanalysis-era5-complete', 
#             download_dict,
#             os.path.join('/Users/joshuamiller/Documents/Lancaster/Data/'+dest_folder, 'ERA5_p='+short_name+'_l='+level+'_2018-04-28_14:00:00'+'.grib'))


















































# def DownloadSST(dest_folder, dates):
    
#     download_dict = {
#         'version': '2_1',
#         'variable': 'all',
#         'format': 'zip',
#         'processinglevel': 'level_4',
#         'sensor_on_satellite': 'combined_product',
#         'year': '',
#         'month': '',
#         'day': ''
#     }
#     #================================================================
#     for date in dates:
#         assert datetime.datetime.strptime(date, "%Y-%m-%d"), date+" has invalid date format. Need YYYY-MM-DD."
#         year, month, day = date.split('-')
#         download_dict['year'] = year
#         download_dict['month'] = month
#         download_dict['day'] = day
      
#         #================================================================
#         c = cdsapi.Client()
#         print("________________________________________________________________________________")
#         print(">> Downloading", 'sst_'+date+'.zip', '\n')
#         c.retrieve('satellite-sea-surface-temperature',
#                 download_dict,
#                 os.path.join(dest_folder, 'sst_'+date+'.zip'))
# DownloadSST(dest_folder="/Users/joshuamiller/Documents/Lancaster/Data",
#             dates=['2018-04-30', '2018-05-01', '2018-05-02'])


'''
This code is based on the one found here:
https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-complete?tab=form

In case the link is broken, here is an example API request given on that webpage:

   #!/usr/bin/env python
   import cdsapi
   c = cdsapi.Client()
   c.retrieve('reanalysis-era5-complete', { # Requests follow MARS syntax
                                            # Keywords 'expver' and 'class' can be dropped. They are obsolete
                                            # since their values are imposed by 'reanalysis-era5-complete'
       'date'    : '2013-01-01',            # The hyphens can be omitted
       'levelist': '1/10/100/137',          # 1 is top level, 137 the lowest model level in ERA5. Use '/' to separate values.
       'levtype' : 'ml',
       'param'   : '130',                   # Full information at https://apps.ecmwf.int/codes/grib/param-db/
                                            # The native representation for temperature is spherical harmonics
       'stream'  : 'oper',                  # Denotes ERA5. Ensemble members are selected by 'enda'
       'time'    : '00/to/23/by/6',         # You can drop :00:00 and use MARS short-hand notation, instead of '00/06/12/18'
       'type'    : 'an',
       'area'    : '80/-50/-25/0',          # North, West, South, East. Default: global
       'grid'    : '1.0/1.0',               # Latitude/longitude. Default: spherical harmonics or reduced Gaussian grid
       'format'  : 'netcdf',                # Output needs to be regular lat-lon, so only works in combination with 'grid'!
   }, 'ERA5-ml-temperature-subarea.nc')     # Output file. Adapt as you wish.
'''