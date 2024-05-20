import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import os

sys.path.append(os.getcwd())
from data_utils.preprocessing_funcs import Scale, UnScale
#====================================================================

data = np.eye(N=10, M=10) * np.random.random((10, 10))

scaled = Scale(data, 'maxabs',
               to_file=True, 
               to_file_path='/Users/joshuamiller/Documents/Python Files/Wildfire_Ozone/data_utils/scale_files',
               data_name='test')

unscaled = UnScale(scaled, 'data_utils/scale_files/test_maxabs.json')

fig, ax = plt.subplots(1, 3, figsize=(8, 4))
fig.subplots_adjust(wspace=.5)

d = ax[0].imshow(data)
s = ax[1].imshow(scaled)
u = ax[2].imshow(unscaled)

ax[0].set_title('Orig')
ax[1].set_title('scaled')
ax[2].set_title('unscaled')

divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(d, cax=cax, label='')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=180)

divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(s, cax=cax, label='')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=180)

divider = make_axes_locatable(ax[2])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(u, cax=cax, label='')
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=180)

plt.show()