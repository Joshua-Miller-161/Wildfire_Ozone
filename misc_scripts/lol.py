import sys
sys.dont_write_bytecode = True
import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import netCDF4 as nc


sys.path.append(os.getcwd())
from misc.plotting_utils import ShowYearMonth, Plot_O3_Fire
from misc.misc_utils import ButterLowpassFilter
from data_utils.extraction_funcs import Extract_netCDF4
#====================================================================
base_path = '/Users/joshuamiller/Documents/Lancaster/Data'

regions=['Whole_Area', 'South_Land', 'North_Land', 'East_Ocean', 'West_Ocean']
regions2=['Whole Area', 'South Land', 'North Land', 'East Ocean', 'West Ocean']

ozone_str = 'S5P_RPRO_L2__O3_TCL'
fire_str  = 'MODIS_C61'

cutoff = 0.016
order = 4

fig, ax = plt.subplots(len(regions), 1, figsize=(20, 10), sharex=True)
fig.subplots_adjust(hspace=0)
#====================================================================
for i in range(len(regions)):

    ozone_file = ''
    fire_file  = ''

    for var in ['Ozone', 'Fire']:
        big_path = os.path.join(os.path.join(base_path, regions[i]), var)
        files = os.listdir(big_path)
        for file in files:
            if file.endswith('.nc'):
                if (ozone_str in file):
                    ozone_file = os.path.join(big_path, file) 
                    #print("ozone_file", ozone_file)
                elif (fire_str in file):
                    fire_file = os.path.join(big_path, file)
                    #print("fire_file", fire_file)

    ozone_dict = Extract_netCDF4(ozone_file,
                                 var_names=['ozone_tropospheric_vertical_column', 'start_date'],
                                 groups='all',
                                 print_sum=False)
    ozone       = np.squeeze(ozone_dict['ozone_tropospheric_vertical_column'])
    ozone_dates = np.squeeze(ozone_dict['start_date'])

    fire_dict = Extract_netCDF4(fire_file,
                                var_names=['frp', 'date'],
                                groups='all',
                                print_sum=False)
    fire       = np.squeeze(fire_dict['frp'])
    fire_dates = np.squeeze(fire_dict['date'])

    #print(np.shape(ozone), np.shape(fire))
    ozone_mean = np.mean(ozone, axis=(1, 2))
    fire_mean  = np.mean(fire, axis=(1, 2))

    ozone_mean = ButterLowpassFilter(ozone_mean, cutoff, 1, order)
    fire_mean  = ButterLowpassFilter(fire_mean, cutoff, 1, order)

    corr = np.corrcoef(fire_mean, ozone_mean)[1][0]
    print(regions[i], corr)

    if (regions[i] == 'North_Land'):
        Plot_O3_Fire(ax[i], fire_dates, fire_mean, ozone_dates, ozone_mean,
                     fire_ylabel='Avg.\ fire\ (MW)',
                     O3_ylabel='Avg.\ ozone\ '+r'\left(\frac{mol}{m^2}\right)',
                     bold=True, fontsize=24, O3_rotation=270, O3_pad=40, fire_xmin=17300, fire_xmax=max(fire_dates)+100)
        
        ax[i].text(17310, .66*max(fire_mean), regions2[i]+'\nCorr: '+str(round(corr, 3)), fontsize=20, fontweight='bold')

    elif (regions[i] == 'West_Ocean'):
        Plot_O3_Fire(ax[i], fire_dates, fire_mean, ozone_dates, ozone_mean,
                     fire_ymin=0, fire_ymax=5, O3_ylabel='', fire_ylabel='',
                     O3_ymin=0.008, O3_ymax=0.024)
        
        ax[i].text(17310, .6*5, regions2[i]+'\nCorr: '+str(round(corr, 3)), fontsize=20, fontweight='bold')

    else:
        if ('Ocean' in regions[i]):
            fire_mean = np.zeros_like(ozone_mean)
            Plot_O3_Fire(ax[i], fire_dates, fire_mean, ozone_dates, ozone_mean,
                        fire_ymin=0, fire_ymax=5, O3_ylabel='', fire_ylabel='')

            ax[i].text(17310, .6*5, regions2[i]+'\nCorr: '+str(round(corr, 3)), fontsize=20, fontweight='bold')

        else:
            Plot_O3_Fire(ax[i], fire_dates, fire_mean, ozone_dates, ozone_mean,
                         O3_ylabel='', fire_ylabel='')
            
            ax[i].text(17310, .66*max(fire_mean), regions2[i]+'\nCorr: '+str(round(corr, 3)), fontsize=20, fontweight='bold')


    ShowYearMonth(ax[i], fire_dates)

    
fig.savefig('/Users/joshuamiller/Documents/Lancaster/Figs/Fire_O3_poster.pdf',
            bbox_inches='tight', pad_inches=0)
#====================================================================
plt.show()