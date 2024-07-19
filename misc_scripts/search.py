import sys
sys.dont_write_bytecode = True
import os
os.environ['USE_PYGEOS'] = '0'
import numpy as np

arr1 = np.squeeze(np.load('/Users/joshuamiller/Documents/Lancaster/SimulationResults/Data/Dense_WA_raw_ozone.npy'))
arr2 = np.squeeze(np.load('/Users/joshuamiller/Documents/Lancaster/SimulationResults/Data/RF_WA_raw_ozone.npy'))

for i in range(np.shape(arr1)[0]):
    for j in range(np.shape(arr2)[0]):
        if (np.array_equal(arr1[i], arr2[j]) == True and (max(arr1[i].ravel()) > 0.03)):
            print("i=", i, ", j=", j, ", eq", np.array_equal(arr1[i], arr2[j]), np.shape(arr1[i]), np.shape(arr2[j]))