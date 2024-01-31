import cdsapi
import pandas as pd
import os
import sys
import datetime

sys.path.append(os.getcwd())
#====================================================================

def DownloadERA5(dest_folder, dates, min_lat, max_lat, min_lon, max_lon, levels, params):
    #================================================================
    for i in range(len(params)):
        params[i] = str(params[i])
    #================================================================
    bbox_str = str(min_lat)+'/'+str(min_lon)+'/'+str(max_lat)+'/'+str(max_lon)
    #================================================================
    levels_str   = ''
    levels_str_f = ''

    for level in levels:
        levels_str += str(int(level)) +'/'
        levels_str_f += str(int(level)) + '-'

    levels_str   = levels_str[:-1] # Get rid of last /
    levels_str_f = levels_str_f[:-1]
    #================================================================
    params_str   = ''
    params_str_f = ''
    for param in params:
        params_str += str(int(param)) + '/'
        params_str_f += str(int(param)) + '-'
    
    params_str   = params_str[:-1]
    params_str_f = params_str_f[:-1]
    #================================================================
    download_dict = {               # Requests follow MARS syntax
                                    # Keywords 'expver' and 'class' can be dropped. They are obsolete
                                    # since their values are imposed by 'reanalysis-era5-complete'
        'date'    : '',             # The hyphens can be omitted
        'levelist': levels_str,     # 1 is top level, 137 the lowest model level in ERA5. Use '/' to separate values.
        'levtype' : 'ml',
        'param'   : params_str,     # Full information at https://apps.ecmwf.int/codes/grib/param-db/
                                    # The native representation for temperature is spherical harmonics
        'stream'  : 'oper',         # Denotes ERA5. Ensemble members are selected by 'enda'
        'time'    : '00/to/23/by/6',# You can drop :00:00 and use MARS short-hand notation, instead of '00/06/12/18'
        'type'    : 'an',
        'area'    : bbox_str,       # North, West, South, East. Default: global
        'grid'    : '.1/.1',      # Latitude/longitude. Default: spherical harmonics or reduced Gaussian grid
        'format'  : 'grib',         # Output needs to be regular lat-lon, so only works in combination with 'grid'!
    }
    #================================================================
    for date in dates:
        assert datetime.datetime.strptime(date, "%Y-%m-%d"), date+" has invalid date format. Need YYYY-MM-DD."

        download_dict['date'] = date
        
        c = cdsapi.Client()
        print("________________________________________________________________________________")
        print(">> Downloading", 'ERA5_p='+params_str_f+'_l='+levels_str_f+'_'+date+'.grib', '\n')
        c.retrieve('reanalysis-era5-complete', 
                   download_dict, 
                   os.path.join(dest_folder, 'ERA5_p='+params_str_f+'_l='+levels_str_f+'_'+date+'.grib'))
        print("\n>> Saved", 'ERA5_p='+params_str_f+'_l='+levels_str_f+'_'+date+'.grib')

#====================================================================
def DownloadSST(dest_folder, dates):
    
    download_dict = {
        'version': '2_1',
        'variable': 'all',
        'format': 'zip',
        'processinglevel': 'level_4',
        'sensor_on_satellite': 'combined_product',
        'year': '',
        'month': '',
        'day': ''
    }
    #================================================================
    for date in dates:
        assert datetime.datetime.strptime(date, "%Y-%m-%d"), date+" has invalid date format. Need YYYY-MM-DD."
        year, month, day = date.split('-')
        download_dict['year'] = year
        download_dict['month'] = month
        download_dict['day'] = day
      
        #================================================================
        c = cdsapi.Client()
        print("________________________________________________________________________________")
        print(">> Downloading", 'sst_'+date+'.zip', '\n')
        c.retrieve('satellite-sea-surface-temperature',
                download_dict,
                os.path.join(dest_folder, 'sst_'+date+'.zip'))
#====================================================================
# DownloadERA5(dest_folder="/Users/joshuamiller/Documents/Lancaster/Data/ERA5",
#              dates=dates,
#              min_lat=20, min_lon=-20, max_lat=-20, max_lon=60,
#              levels=[1,137],
#              params=[131, 132])

DownloadSST(dest_folder="/Users/joshuamiller/Documents/Lancaster/Data",
            dates=['2018-04-30', '2018-05-01', '2018-05-02'])











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