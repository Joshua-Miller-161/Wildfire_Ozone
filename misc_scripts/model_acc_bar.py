import sys
sys.dont_write_bytecode = True
import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
#====================================================================
fig, ax = plt.subplots(1, 1, figsize = (12, 6))

ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
#====================================================================
colors = ['#C0504D', '#488DC8', '#9BBB59', '#8064A2', '#F79646']

df = pd.read_excel("/Users/joshuamiller/Documents/Lancaster/Poster/ModelErrorPoster.xlsx", index_col=0)

regions = ['Whole Area', 'South Land', 'North Land', 'East Ocean', 'West Ocean']
print(df)
#====================================================================
models  = df.columns

bar_width = 0.1

x = np.arange(len(regions))

for i in range(len(regions)):
    if (i == 0):
        for j in range(len(models)):
            ax.bar(x[i] - 0.5 * (len(models)*bar_width) + (j * bar_width), df.loc[regions[i], models[j]], bar_width, label=models[j], color=colors[j])
    else:
        for j in range(len(models)):
            ax.bar(x[i] - 0.5 * (len(models)*bar_width) + (j * bar_width), df.loc[regions[i], models[j]], bar_width, color=colors[j])


regions2 = regions = ['Whole\nArea', 'South\nLand', 'North\nLand', 'East\nOcean', 'West\nOcean']

ax.set_ylabel('Model error (MSE)', fontweight='bold', fontsize=18)
ax.set_xticks(x, regions2, fontweight='bold')  # Set x-axis labels
ax.grid(axis='y')
ax.legend()

# Show the plot
plt.tight_layout()  # Ensure labels are visible
plt.show()
#====================================================================