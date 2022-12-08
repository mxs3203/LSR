
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import rfftfreq
from scipy.signal import find_peaks

total_data = pd.DataFrame([])

def fft_for_curve(curve, f_ratio=1, DURATION=201):
    yf = np.fft.fft(curve)
    freq = np.linspace(0, 1, len(yf))
    num_freq_bins = int(len(freq) * f_ratio)
    plt.plot(freq[:num_freq_bins], yf[:num_freq_bins]/DURATION)
    plt.xlabel("Freq (Hz)")
    plt.show()
    return np.abs(yf[:num_freq_bins])/DURATION


for f in glob.glob("/home/mateo/LSR/example_database/train_data/*.pickle"):
    with open(f, 'rb') as file:
        a = pd.read_pickle(f)
        curve = a.curve.data_frame['value'].values
        curve[curve < 0] = 0
        que = curve[find_peaks(curve, height=1, threshold=1)[0]]
        extracted_fft = que[0:9]#fft_for_curve(curve, f_ratio=0.2)
        row = a.ten_nums
        row.append(-1)
        row.extend(extracted_fft)
        row.extend(curve)
        total_data = pd.concat([total_data, pd.DataFrame(row).transpose()], ignore_index=True)

print(total_data)
total_data.to_csv("input_data_with_fft.csv")