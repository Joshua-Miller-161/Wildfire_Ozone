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

df_mse_old  = pd.read_excel(os.path.join(path, mse+".xlsx"), sheet_name='OFTUVXYD', index_col=0)
df_madn_old = pd.read_excel(os.path.join(path, madn+".xlsx"), sheet_name='OFTUVXYD', index_col=0)
df_hot_old  = pd.read_excel(os.path.join(path, perc+".xlsx"), sheet_name='OFTUVXYD', index_col=0)

df_mse_new  = pd.read_excel(os.path.join(path, mse+".xlsx"), sheet_name='OTUVXYD', index_col=0)
df_madn_new = pd.read_excel(os.path.join(path, madn+".xlsx"), sheet_name='OTUVXYD', index_col=0)
df_hot_new  = pd.read_excel(os.path.join(path, perc+".xlsx"), sheet_name='OTUVXYD', index_col=0)
#====================================================================
ranked_mse_old = df_mse_old.apply(rank_columns, axis=1, args=(False,))

ranked_madn_old = df_madn_old.apply(rank_columns, axis=1, args=(False,))

ranked_hot_old = df_hot_old.apply(rank_columns, axis=1, args=(True,))

ranked_mse_new = df_mse_new.apply(rank_columns, axis=1, args=(False,))

ranked_madn_new = df_madn_new.apply(rank_columns, axis=1, args=(False,))

ranked_hot_new = df_hot_new.apply(rank_columns, axis=1, args=(True,))

total_abs_df = abs(ranked_mse_old - ranked_mse_new) + abs(ranked_madn_old - ranked_madn_new) + abs(ranked_hot_old - ranked_hot_new)

total_df = (ranked_mse_old - ranked_mse_new) + (ranked_madn_old - ranked_madn_new) + (ranked_hot_old - ranked_hot_new)
#====================================================================
print(df_mse_old)
print("=========================")
print(ranked_mse_old)
print("=========================")
print(ranked_mse_new)
print("===========================================================================")
print("===========================================================================")
print("===========================================================================")
print(abs(ranked_mse_old - ranked_mse_new))
print("=========================")
print(abs(ranked_madn_old - ranked_madn_new))
print("=========================")
print(abs(ranked_hot_old - ranked_hot_new))
print("===========================================================================")
print("===========================================================================")
print("===========================================================================")
print(total_abs_df)
print("===========================================================================")
print("===========================================================================")
print("===========================================================================")
print(total_abs_df.sum(axis=0))
































# #====================================================================
# total_rankings = ranked_mse_old.sum(axis=0) + ranked_madn_old.sum(axis=0) + ranked_perc_old.sum(axis=0)
# print("____________________________________________________________")
# print(" - > All regions all metrics")
# print(list(df_mse_old.columns))
# print(list(total_rankings))
# Sort(list(df_mse_old.columns), list(total_rankings))
# print("____________________________________________________________")
# print(" - > All regions MSE")
# print(list(df_mse_old.columns))
# print(list(ranked_mse_old.sum(axis=0)))
# Sort(list(df_mse_old.columns), list(ranked_mse_old.sum(axis=0)))
# print("____________________________________________________________")
# print(" - > All regions MADN")
# print(list(df_madn_old.columns))
# print(list(ranked_madn_old.sum(axis=0)))
# Sort(list(df_madn_old.columns), list(ranked_madn_old.sum(axis=0)))
# print("____________________________________________________________")
# print(" - > All regions hotspot")
# print(list(df_perc_old.columns))
# print(list(ranked_perc_old.sum(axis=0)))
# Sort(list(df_perc_old.columns), list(ranked_perc_old.sum(axis=0)))
# print("____________________________________________________________")
# print(" ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^")
# print("^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^")
# print(" ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^ ^_^")
# print("____________________________________________________________")
# #====================================================================
# dict_ = {'MSE': ranked_mse_old, 'MADN': ranked_madn_old, 'Hotspot': ranked_perc_old}
# for region in regions:
#     sum_rankings = ranked_mse_old.loc[region] + ranked_madn_old.loc[region] + ranked_perc_old.loc[region]
#     print(" - >", region, " all metrics")
#     print(list(ranked_mse_old.columns))
#     print(list(sum_rankings))
#     Sort(list(ranked_mse_old.columns), list(sum_rankings))
#     print("____________________________________________________________")