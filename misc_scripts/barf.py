import sys
sys.dont_write_bytecode = True
import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.getcwd())
from misc.plotting_utils import ShowYearMonth

def ParseModelName(input_string, substrs=['reg=', 'In=', 'Out=', 'e='], split_char='_'):

    dict_ = {'WA': 'Whole Area',
             'EO': 'East Ocean',
             'WO': 'West Ocean', 
             'SL': 'South Land',
             'NL': 'North Land',
             'RF': 'Random Forest',
             'Trans': 'Transformer'}

    info = []
    start = len(input_string.split(split_char)[0]) + 1

    info.append(dict_[input_string.split(split_char)[0]])

    print("info:", info, input_string)

    for substr in substrs:
        temp_str = input_string[start:]
        if (substr in temp_str):

            lol = temp_str.split(split_char)[0]
            lol = lol[len(substr):]
            start += len(substr) + len(lol) + 1
            print(temp_str, substr, lol)

            try:
                info.append(dict_[lol])
            except KeyError:
                info.append(lol)

    return info
# Example usage:
result = ParseModelName("RF_reg=WA_In=LOL_Out=X_e=10.json")
print(result) 