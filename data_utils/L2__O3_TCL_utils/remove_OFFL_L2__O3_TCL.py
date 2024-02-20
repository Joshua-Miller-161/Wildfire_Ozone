import os
import sys
import shutil

sys.path.append(os.getcwd())
from misc.misc_utils import GetDateInStr

folder_path = ['/Users/joshuamiller/Documents/Lancaster/Data/Kriged_L2_O3_TCL',
               '/Users/joshuamiller/Documents/Lancaster/Data/East_Ocean/L2__O3_TCL',
               '/Users/joshuamiller/Documents/Lancaster/Data/West_Ocean/L2__O3_TCL',
               '/Users/joshuamiller/Documents/Lancaster/Data/North_Land/L2__O3_TCL',
               '/Users/joshuamiller/Documents/Lancaster/Data/South_Land/L2__O3_TCL']

for folder in folder_path:
    RPRO_files = []
    OFFL_files = []
    for file in os.listdir(folder):
        if (file.endswith('.nc') or file.endswith('.csv')):
            if 'RPRO' in file:
                RPRO_files.append(os.path.join(folder, file))
            elif 'OFFL' in file:
                OFFL_files.append(os.path.join(folder, file))

    for rpro_file in RPRO_files:
        rpro_date = GetDateInStr(rpro_file)
        i = 0
        while (i < len(OFFL_files)):
            offl_date = GetDateInStr(OFFL_files[i])
            if (rpro_date == offl_date):
                print("i=",i,"MATCH:\n", rpro_file,'\n',OFFL_files[i])
                print("==================================")
                shutil.move(OFFL_files[i], "/Users/joshuamiller/.Trash")
            i += 1
    print("++++++++++++++++++++++++++++++++++++++++++++++")