
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

total_data = pd.DataFrame([])
EPS = 0.0001

def find_values_by_color(curve_df):
    min_nm = min(curve_df['nm'])
    max_nm = max(curve_df['nm'])
    curve_df['value'] = np.log10(curve_df['value']+EPS)
    violet = curve_df[(curve_df['nm'] >= min_nm) & (curve_df['nm'] <= 400)]
    violet = violet['value'].mean()
    blue = curve_df[(curve_df['nm'] >= 400) & (curve_df['nm'] <= 450)]
    blue = blue['value'].mean()
    cyan = curve_df[(curve_df['nm'] >= 450) & (curve_df['nm'] <= 500)]
    cyan = cyan['value'].mean()
    green = curve_df[(curve_df['nm'] >= 500) & (curve_df['nm'] <= 550)]
    green = green['value'].mean()
    yellow = curve_df[(curve_df['nm'] >= 550) & (curve_df['nm'] <= 580)]
    yellow = yellow['value'].mean()
    orange = curve_df[(curve_df['nm'] >= 580) & (curve_df['nm'] <= 600)]
    orange = orange['value'].mean()
    red = curve_df[(curve_df['nm'] >= 600) & (curve_df['nm'] <= 650)]
    red = red['value'].mean()
    far_red = curve_df[(curve_df['nm'] >= 650) & (curve_df['nm'] <= 700)]
    far_red = far_red['value'].mean()
    after_red = curve_df[(curve_df['nm'] >= 700) & (curve_df['nm'] <= max_nm)]
    after_red = after_red['value'].mean()
    return [violet, blue, cyan, green, yellow, orange, red, far_red, after_red]

for f in glob.glob("/home/mateo/LSR-main/example_database/train_data/*.pickle"):
    with open(f, 'rb') as file:
        a = pd.read_pickle(f)
        a.ten_nums = np.array(a.ten_nums)
        a.ten_nums = np.log10(a.ten_nums + EPS)
        curve = a.curve.data_frame['value'].values
        curve[curve < 0] = 0
        curve = np.log10(curve+EPS)
        curve = curve
        row = a.ten_nums
        #row = np.append(row, -1)
        row = np.append(row, curve)
        total_data = pd.concat([total_data, pd.DataFrame(row).transpose()], ignore_index=True)

print(total_data)
total_data.to_csv("~/LSR-main/modeling/input_data_with_fft.csv")