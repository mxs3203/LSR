
import glob
from numpy.fft import fft, ifft
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
total_data = pd.DataFrame([])

def fft_for_curve(curve, sr = 201, nummber_of_freqs=30):
    fft_transform = fft(curve)
    N = len(curve)
    n = np.arange(N)
    T = N / sr
    ts = 1.0 / sr
    t = np.arange(0, 1, ts)
    freq = n / T

    # plt.figure(figsize=(12, 6))
    # plt.subplot(121)
    # plt.stem(freq, np.abs(fft_transform), 'b', markerfmt=" ", basefmt="-b")
    # plt.xlabel('Freq (Hz)')
    # plt.ylabel('FFT Amplitude |X(freq)|')
    # plt.xlim(0, nummber_of_freqs)
    # plt.subplot(122)
    # plt.plot(t, ifft(fft_transform), 'r')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.tight_layout()
    # plt.show()
    return np.abs(fft_transform)[0:nummber_of_freqs]

for f in glob.glob("/home/mateo/LSR/example_database/train_data/*.pickle"):
    with open(f, 'rb') as file:
        a = pd.read_pickle(f)
        curve_file = a.file_name
        file_name = curve_file.split("/")[1]
        with open("/home/mateo/LSR/example_database/" + file_name.upper(), 'rb') as f2:
            curve = pd.read_csv(f2, delimiter=" ", skiprows=1, names=['nm', 'ignore', 'value'])
            curve = curve.loc[(curve['nm'] >= 350) & (curve['nm'] <= 850)]
            curve = curve.groupby(np.arange(len(curve)) // 5).agg({"nm":'mean','value': 'mean'})
            curve[curve < 0] = 0
            extracted = fft_for_curve(curve['value'].values)

        row = a.ten_nums
        row.append(-1)
        row.extend(extracted)
        row.extend(curve['value'].values)
        total_data = pd.concat([total_data, pd.DataFrame(row).transpose()], ignore_index=True)

print(total_data)

total_data.to_csv("input_data_with_fft.csv")