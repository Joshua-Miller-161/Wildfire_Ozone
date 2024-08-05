import sys
sys.dont_write_bytecode = True
import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
from matplotlib.colors import Normalize, LogNorm, FuncNorm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as font_manager
import pandas as pd
import fiona
import numpy as np
from datetime import datetime, timedelta
import yaml

sys.path.append(os.getcwd())
from ml.ml_utils import ParseModelName
from misc.misc_utils import GetBoxCoords, PercentChange
#====================================================================
def Sort(strings, numbers):
    # Combine the lists into a list of tuples
    combined = list(zip(numbers, strings))

    # Sort the combined list by the numerical values in descending order
    combined_sorted = sorted(combined, key=lambda x: x[0], reverse=True)

    # Unzip the sorted list back into two lists
    numbers_sorted, strings_sorted = zip(*combined_sorted)

    # Convert the tuples back to lists
    numbers_sorted = list(numbers_sorted)
    strings_sorted = list(strings_sorted)

    print(numbers_sorted)
    print(strings_sorted)
#====================================================================
''' Get data'''
base_path = "/Users/joshuamiller/Documents/Lancaster/Dissertation"
filename = "Final_MSE"

df_old_mse = pd.read_excel(os.path.join(base_path, os.path.join('Data', "Final_MSE.xlsx")),
                      sheet_name='OFTUVXYD', index_col=0)
df_new_mse = pd.read_excel(os.path.join(base_path, os.path.join('Data', "Final_MSE.xlsx")),
                      sheet_name='OTUVXYD', index_col=0)

df_old_madn = pd.read_excel(os.path.join(base_path, os.path.join('Data', "FinalMSE_MADN.xlsx")),
                      sheet_name='OFTUVXYD', index_col=0)
df_new_madn = pd.read_excel(os.path.join(base_path, os.path.join('Data', "FinalMSE_MADN.xlsx")),
                      sheet_name='OTUVXYD', index_col=0)

df_old_hot = pd.read_excel(os.path.join(base_path, os.path.join('Data', "Final99Percent.xlsx")),
                      sheet_name='OFTUVXYD', index_col=0)
df_new_hot = pd.read_excel(os.path.join(base_path, os.path.join('Data', "Final99Percent.xlsx")),
                      sheet_name='OTUVXYD', index_col=0)

perc_diff_mse = 100 * (df_new_mse - df_old_mse) / df_old_mse

perc_diff_madn = 100 * (df_new_madn - df_old_madn) / df_old_madn

perc_diff_hot = 100 * (df_new_hot - df_old_hot) / df_old_hot

df_list = [perc_diff_mse, perc_diff_madn, perc_diff_hot]

#print(perc_diff_hot.loc['North Land', :], perc_diff_hot.loc['East Ocean', :])

print(" >> mse sum:", perc_diff_mse.sum().sum() / 50)
print(" >> madn sum:", perc_diff_madn.sum().sum() / 50)
print(" >> hot sum:", perc_diff_hot.sum().sum() / 50)


# print(" >> std sum:", perc_diff_hot.stack().std())
# perc_diff_hot[perc_diff_hot > 130] = 0
# print(" >> hot sum:", perc_diff_hot.sum().sum() / 48)

#print(perc_diff_madn.loc[:, 'GBM'], perc_diff_madn.loc[:, 'GBM'].sum()/5)
#====================================================================
''' Initialize plot '''
fig, ax = plt.subplots(3, 1, figsize = (12, 8), sharex=True)
fig.subplots_adjust(hspace=0)

colors = ['darkred', 'lightcoral', 
          'purple', 'violet',
          'goldenrod', 'gold',
          'navy', 'deepskyblue',
          'forestgreen', 'lawngreen']

regions = ['Whole Area', 'South Land', 'North Land', 'East Ocean', 'West Ocean']
models = df_old_mse.columns

bar_width = 0.075
x = np.arange(len(regions))

# uh = []
# for model in models:
#     sum_mse  = perc_diff_mse.loc[:, model].sum()
#     sum_madn = perc_diff_madn.loc[:, model].sum()
#     sum_hot  = perc_diff_hot.loc[:, model].sum()
#     uh.append((sum_mse + sum_madn + sum_hot) / 15)
#     print(" >>", model, round((sum_mse + sum_madn + sum_hot) / 15, 3))
# print("---------------------------------------------------------")
# print(Sort(models, uh))
# print("---------------------------------------------------------")


print(" >>", np.corrcoef([[2, 1, 8,10, 7, 5, 9, 6, 4, 3],
                          [1, 2, 3, 4, 5, 6, 7, 8, 9,10]]))

print(" >>", np.corrcoef([[3, 1, 8, 9, 7, 2, 10,4, 5, 6],
                          [1, 2, 3, 4, 5, 6, 7, 8, 9,10]]))
#====================================================================
''' Plot '''
for h in range(3):
    for i in range(len(regions)):
        if (i == 0):
            for j in range(len(models)):
                if ("Trans." in models[j]):
                    ax[h].bar(x[i] - 0.5 * (len(models)*bar_width) + (j * bar_width), df_list[h].loc[regions[i], models[j]], bar_width, label=models[j], color=colors[j], hatch='x', edgecolor='black')
                else:
                    ax[h].bar(x[i] - 0.5 * (len(models)*bar_width) + (j * bar_width), df_list[h].loc[regions[i], models[j]], bar_width, label=models[j], color=colors[j], edgecolor='black')
        else:
            for j in range(len(models)):
                if ("Trans." in models[j]):
                    ax[h].bar(x[i] - 0.5 * (len(models)*bar_width) + (j * bar_width), df_list[h].loc[regions[i], models[j]], bar_width, color=colors[j], hatch='x', edgecolor='black')
                else:
                    ax[h].bar(x[i] - 0.5 * (len(models)*bar_width) + (j * bar_width), df_list[h].loc[regions[i], models[j]], bar_width, color=colors[j], edgecolor='black')
#====================================================================
''' Format '''
regions2 = ['Whole\nArea', 'South\nLand', 'North\nLand', 'East\nOcean', 'West\nOcean']

ax[1].set_ylabel('Performance Change (%)', fontweight='bold', fontsize=22)

ax[0].set_yticks([-20, -10, 0, 10, 20])
ax[0].set_ylim(-25, 40)

ax[1].set_yticks([-30, -20, -10, 0, 10, 20, 30])
ax[1].set_ylim(-35, 35)

ax[2].set_yticks(np.arange(-100, 110, 20))
ax[2].set_ylim(-110, 115)

ax_mse_label     = ax[0].twinx()
ax_madn_label    = ax[1].twinx()
ax_hotspot_label = ax[2].twinx()
ax_mse_label.set_ylabel('MSE', fontweight='bold', fontsize=16, rotation=270, labelpad=20)
ax_mse_label.set_yticks([])
ax_madn_label.set_ylabel('MADN', fontweight='bold', fontsize=16, rotation=270, labelpad=20)
ax_madn_label.set_yticks([])
ax_hotspot_label.set_ylabel('Hotspots', fontweight='bold', fontsize=16, rotation=270, labelpad=20)
ax_hotspot_label.set_yticks([])

for i in range(3):
    ax[i].grid(axis='y', which='major', linestyle='-')
    ax[i].grid(axis='y', which='minor', linestyle=':')
    ax[i].tick_params(axis='y', which='major', labelsize=15)

ax[2].set_xticks(x-(1 / (8*np.shape(x)[0])), regions2, fontweight='bold', fontsize=17)

font = font_manager.FontProperties(weight='bold',  # family='Comic Sans MS', 'Times new roman', ,
                                   style='normal', size=11)
ax[0].legend(loc='upper center', prop=font, ncol=5)

#plt.tight_layout()  # Ensure labels are visible
#====================================================================
plt.show()
#====================================================================
#fig.savefig(os.path.join(base_path, os.path.join('Figs', 'Fire_noFire_Percent.pdf')), bbox_inches='tight', pad_inches=0)