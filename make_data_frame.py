
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

total_data = pd.DataFrame([])
EPS = 0.0001


def fft_for_curve(curve, f_ratio=1, DURATION=201):
    yf = np.fft.fft(curve)
    freq = np.linspace(0, 1, len(yf))
    num_freq_bins = int(len(freq) * f_ratio)
    plt.plot(freq[:num_freq_bins], yf[:num_freq_bins]/DURATION)
    plt.xlabel("Freq (Hz)")
    plt.show()
    return np.abs(yf[:num_freq_bins])/DURATION

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

def findLSRPeaks(curve_df):
    first = curve_df.loc[(curve_df['nm'] >= 363) & (curve_df['nm'] <= 376), 'value'].mean()
    second = curve_df.loc[(curve_df['nm'] >= 383) & (curve_df['nm'] <= 396), 'value'].mean()
    third = curve_df.loc[(curve_df['nm'] >= 456) & (curve_df['nm'] <= 468), 'value'].mean()
    fourth = curve_df.loc[(curve_df['nm'] >= 436) & (curve_df['nm'] <= 446), 'value'].mean()
    fifth = curve_df.loc[(curve_df['nm'] >= 516) & (curve_df['nm'] <= 536), 'value'].mean()
    six = curve_df.loc[(curve_df['nm'] >= 441) & (curve_df['nm'] <= 448), 'value'].mean()
    six2 = curve_df.loc[(curve_df['nm'] >= 568) & (curve_df['nm'] <= 638), 'value'].mean()
    seven = curve_df.loc[(curve_df['nm'] >= 591) & (curve_df['nm'] <= 596), 'value'].mean()
    eight = curve_df.loc[(curve_df['nm'] >= 626) & (curve_df['nm'] <= 631), 'value'].mean()
    nine = curve_df.loc[(curve_df['nm'] >= 653) & (curve_df['nm'] <= 661), 'value'].mean()
    ten = curve_df.loc[(curve_df['nm'] >= 728) & (curve_df['nm'] <= 741), 'value'].mean()
    return np.array([first,second, third,fourth, fifth, six, six2,seven,eight,nine,ten])

for f in glob.glob("example_database/train_data/*.pickle"):
    with open(f, 'rb') as file:
        a = pd.read_pickle(f)
        a.ten_nums = np.array(a.ten_nums)
        a.ten_nums = np.log10(a.ten_nums + EPS)
        data_frame = pd.DataFrame(pd.DataFrame.transpose(a.curve.data_frame).T)
        data_frame = data_frame.loc[(data_frame['nm'] >= 350) & (data_frame['nm'] <= 750)]
        data_frame = data_frame.groupby(np.arange(len(data_frame)) // 5).agg({"nm": 'mean', 'value': 'mean'})
        lsr_peaks = findLSRPeaks(data_frame)
        #lsr_peaks = np.log10(lsr_peaks + EPS)
        curve = data_frame['value'].values
        curve[curve < 0] = 0
        #curve = np.log10(curve+EPS)
        row = a.ten_nums
        row = np.append(row, [-10])
        row = np.append(row, lsr_peaks)
        row = np.append(row, [-10])
        row = np.append(row, curve)
        total_data = pd.concat([total_data, pd.DataFrame(row).transpose()], ignore_index=True)

print(total_data)
total_data.to_csv("modeling/input_data_with_fft.csv")
