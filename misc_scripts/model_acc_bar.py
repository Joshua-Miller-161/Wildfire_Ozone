import sys
sys.dont_write_bytecode = True
import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.font_manager as font_manager
import numpy as np
import pandas as pd
#====================================================================
fig, ax = plt.subplots(1, 1, figsize = (12, 5))

ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
#====================================================================
colors = ['darkred', 'lightcoral', 
          'purple', 'violet',
          'goldenrod', 'gold',
          'navy', 'deepskyblue',
          'forestgreen', 'lawngreen']

path = "/Users/joshuamiller/Documents/Lancaster/Dissertation"
name = "Final99Percent"

df = pd.read_excel(os.path.join(path, name+".xlsx"), index_col=0)

regions = ['Whole Area', 'South Land', 'North Land', 'East Ocean', 'West Ocean']
print(df)
#====================================================================
models  = df.columns

bar_width = 0.05

x = np.arange(len(regions))

# for i in range(len(regions)):
#     if (i == 0):
#         for j in range(len(models)):
#             ax.bar(x[i] - 0.5 * (len(models)*bar_width) + (j * bar_width), df.loc[regions[i], models[j]], bar_width, label=models[j], color=colors[j])
#     else:
#         for j in range(len(models)):
#             ax.bar(x[i] - 0.5 * (len(models)*bar_width) + (j * bar_width), df.loc[regions[i], models[j]], bar_width, color=colors[j])


for i in range(len(regions)):
    if (i == 0):
        for j in range(len(models)):
            if ("Trans." in models[j]):
                ax.bar(x[i] - 0.5 * (len(models)*bar_width) + (j * bar_width), df.loc[regions[i], models[j]], bar_width, label=models[j], color=colors[j], hatch='x', edgecolor='black')
            else:
                ax.bar(x[i] - 0.5 * (len(models)*bar_width) + (j * bar_width), df.loc[regions[i], models[j]], bar_width, label=models[j], color=colors[j], edgecolor='black')
    else:
        for j in range(len(models)):
            if ("Trans." in models[j]):
                ax.bar(x[i] - 0.5 * (len(models)*bar_width) + (j * bar_width), df.loc[regions[i], models[j]], bar_width, color=colors[j], hatch='x', edgecolor='black')
            else:
                ax.bar(x[i] - 0.5 * (len(models)*bar_width) + (j * bar_width), df.loc[regions[i], models[j]], bar_width, color=colors[j], edgecolor='black')

regions2 = ['Whole\nArea', 'South\nLand', 'North\nLand', 'East\nOcean', 'West\nOcean']

ax.set_ylabel('Avg. Correct Hotspots', fontweight='bold', fontsize=22)
ax.set_xticks(x-(1 / (8*np.shape(x)[0])), regions2, fontweight='bold', fontsize=17)  # Set x-axis labels
ax.set_ylim(10**-5, 1.3 * 10**-2)
ax.set_yscale('log')
ax.grid(axis='y', which='major', linestyle='-')
ax.grid(axis='y', which='minor', linestyle=':')
ax.tick_params(axis='y', which='major', labelsize=15)

font = font_manager.FontProperties(weight='bold',  # family='Comic Sans MS', 'Times new roman', ,
                                   style='normal', size=12)
ax.legend(loc='upper center', prop=font, ncol=5)

# Show the plot
plt.tight_layout()  # Ensure labels are visible
plt.show()
#====================================================================
fig.savefig(os.path.join(path, name+'.pdf'), bbox_inches='tight', pad_inches=0)
#====================================================================