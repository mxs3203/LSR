
import glob
import pickle

import pandas as pd


total_data = pd.DataFrame()

with open("/home/mateo/Desktop/LSR_data/train_data/obs_5.pickle", 'rb') as f:
    a = pd.read_pickle(f)
    curve_file = a.file_name
    file_name = curve_file.split("/")[1]
    with open("/home/mateo/Desktop/LSR_data/SSM_data/" + file_name.upper(), 'rb') as f2:
        curve = pd.read_csv(f2,delimiter=" ",skiprows=1, names=['nm','ignore','value'])
        curve = curve.loc[(curve['nm'] >= 350) & (curve['nm'] <= 850)]
        curve = curve.iloc[::5, :]
    row = a.ten_nums
    row.append(-1)
    row.extend(curve['value'].values)
    total_data = total_data.append([row], ignore_index=True)

for f in glob.glob("/home/mateo/Desktop/LSR_data/train_data/*.pickle"):
    with open(f, 'rb') as file:
        a = pd.read_pickle(f)
        curve_file = a.file_name
        file_name = curve_file.split("/")[1]
        with open("/home/mateo/Desktop/LSR_data/SSM_data/" + file_name.upper(), 'rb') as f2:
            curve = pd.read_csv(f2, delimiter=" ", skiprows=1, names=['nm', 'ignore', 'value'])
            curve = curve.loc[(curve['nm'] >= 350) & (curve['nm'] <= 850)]
            curve = curve.iloc[::5, :]
        row = a.ten_nums
        row.append(-1)
        row.extend(curve['value'].values)
        total_data = total_data.append([row], ignore_index=True)

print(total_data)

total_data.to_csv("input_data.csv")