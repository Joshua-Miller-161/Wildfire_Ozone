import sys
sys.dont_write_bytecode = True
import pandas as pd
import numpy as np
import os
#====================================================================
def rank_columns(row, ascending=False):
    return row.rank(ascending=ascending, method='min').astype(int)

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
regions = ['Whole Area', 'East Ocean', 'West Ocean', 'South Land', 'North Land']

path = "/Users/joshuamiller/Documents/Lancaster/Dissertation/Data"

mse  = "Final_MSE"
madn = "FinalMSE_MADN"
perc = "Final99Percent"

df_mse  = pd.read_excel(os.path.join(path, mse+".xlsx"), sheet_name='OFTUVXYD', index_col=0)
df_madn = pd.read_excel(os.path.join(path, madn+".xlsx"), sheet_name='OFTUVXYD', index_col=0)
df_perc = pd.read_excel(os.path.join(path, perc+".xlsx"), sheet_name='OFTUVXYD', index_col=0)
#====================================================================
ranked_mse = df_mse.apply(rank_columns, axis=1, args=(False,))

ranked_madn = df_madn.apply(rank_columns, axis=1, args=(False,))

ranked_perc = df_perc.apply(rank_columns, axis=1, args=(True,))
#====================================================================
# print(df_mse)
# print("=========================")
# print(ranked_mse)


#====================================================================
total_rankings = ranked_mse.sum(axis=0) + ranked_madn.sum(axis=0) + ranked_perc.sum(axis=0)
print("____________________________________________________________")
print(" - > All regions all metrics")
print(list(df_mse.columns))
print(list(total_rankings))
Sort(list(df_mse.columns), list(total_rankings))
print("____________________________________________________________")
print(" - > All regions MSE")
print(list(df_mse.columns))
print(list(ranked_mse.sum(axis=0)))
Sort(list(df_mse.columns), list(ranked_mse.sum(axis=0)))
print("____________________________________________________________")
print(" - > All regions MADN")
print(list(df_madn.columns))
print(list(ranked_madn.sum(axis=0)))
Sort(list(df_madn.columns), list(ranked_madn.sum(axis=0)))
print("____________________________________________________________")
print(" - > All regions hotspot")
print(list(df_perc.columns))
print(list(ranked_perc.sum(axis=0)))
Sort(list(df_perc.columns), list(ranked_perc.sum(axis=0)))
print("____________________________________________________________")
print(" ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^")
print("^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^")
print(" ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^")
print("____________________________________________________________")
#====================================================================
dict_ = {'MSE': ranked_mse, 'MADN': ranked_madn, 'Hotspot': ranked_perc}
for region in regions:
    sum_rankings = ranked_mse.loc[region] + ranked_madn.loc[region] + ranked_perc.loc[region]
    print(" - >", region, " all metrics")
    print(list(ranked_mse.columns))
    print(list(sum_rankings))
    Sort(list(ranked_mse.columns), list(sum_rankings))
    print("____________________________________________________________")