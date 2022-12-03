import os
import time

from numpy.fft import fft, ifft

import matplotlib.pyplot as plt
from LSR_comm import LSR_comm
from SpectraWizSaver import save_curve
from modeling.Curve_Dataloader import Curve_Loader
from modeling.Predict10 import Predict10
import torch
import pandas as pd
import numpy as np

loader = Curve_Loader("~/LSR/modeling/input_data_with_fft.csv",fft_size=30)
scaler = loader.getScaler()

input_curve_file = "/home/mateo/LSR/ZAGREB071022/Akrozprozor.ssm"
#lsr = LSR_comm("COM3")
time.sleep(1)  # waiting for automation start

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Predict10(curve_size=30)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()


def fft_for_curve(curve, sr=201, nummber_of_freqs=30):
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

def readAndCurateCurve(file, scaler):
    with open(file, 'rb') as f2:
        curve = pd.read_csv(f2, delimiter=" ", skiprows=1, names=['nm', 'ignore', 'value'])
        curve = curve.loc[(curve['nm'] >= 350) & (curve['nm'] <= 850)]
        curve = curve.groupby(np.arange(len(curve)) // 5).agg({"nm": 'mean', 'value': 'mean'})
        curve[curve < 0] = 0
        extracted = pd.DataFrame(fft_for_curve(curve['value'].values))
        extracted = scaler.fit_transform(extracted).transpose()
        return extracted, curve

extracted,curve = readAndCurateCurve(input_curve_file, scaler)
predicted_ten_nums = model(torch.FloatTensor(extracted))
predicted_ten_nums = predicted_ten_nums.detach().numpy().squeeze()
predicted_ten_nums = [1.0 if ele > 1.0 else ele for ele in predicted_ten_nums]
print(predicted_ten_nums)
# lsr.set_column_data(1, predicted_ten_nums)
# lsr.set_column_data(2, lsr.compute_column_based_on_first(0.7))
# lsr.set_column_data(3, lsr.compute_column_based_on_first(0.5))
# lsr.set_column_data(4, lsr.compute_column_based_on_first(0.3))
# lsr.run()  # this take 1 sec
# Lets wait for the curve
time.sleep(1)
# saving takes 0.3sec
save_curve("{}".format(input_curve_file + "_recreated"))
print("Waiting for recreated file to be saved...")
while not os.path.exists("example_database/{}.ssm".format(input_curve_file + "_recreated")):
    time.sleep(1)
extracetd_reconstructed, reconstructed_curve = readAndCurateCurve("example_database/{}.ssm".format(input_curve_file + "_recreated"), scaler)

mse = ((reconstructed_curve - curve)**2).mean(axis=ax)
print(mse)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(range(201), curve, 'b', markerfmt=" ", basefmt="-b")
plt.subplot(122)
plt.plot(range(201), reconstructed_curve, 'r')
plt.tight_layout()
plt.show()


