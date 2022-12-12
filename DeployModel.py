import os
import time

import matplotlib.pyplot as plt

import admin
from LSR_comm import LSR_comm
from SpectraWizSaver import save_curve
from modeling.Curve_Dataloader import Curve_Loader
from modeling.Predict10 import Predict10
import torch
import pandas as pd
import numpy as np

loader = Curve_Loader(r"modeling/input_data_with_fft.csv", fft_size=30)
scalerX = loader.getScalerX()
scalerY = loader.getScalerY()

input_curve_file = r"example_database/test_soba.ssm"
sample_name = input_curve_file.split("/")
sample_name = sample_name[len(sample_name)-1]
sample_name = sample_name.split(".")[0]
lsr = LSR_comm("COM3")
time.sleep(1)  # waiting for automation start

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Predict10(curve_size=40)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

def fft_for_curve(curve, f_ratio=1, DURATION=201):
    yf = np.fft.fft(curve)
    freq = np.linspace(0, 1, len(yf))
    num_freq_bins = int(len(freq) * f_ratio)
    plt.plot(freq[:num_freq_bins], yf[:num_freq_bins]/DURATION)
    plt.xlabel("Freq (Hz)")
    plt.show()
    return np.abs(yf[:num_freq_bins])/DURATION


def readAndCurateCurve(file, scaler):
    with open(file, 'rb') as f2:
        curve = pd.read_csv(f2, delimiter=" ", skiprows=1, names=['nm', 'ignore', 'value'])
        curve = curve.loc[(curve['nm'] >= 350) & (curve['nm'] <= 850)]
        curve = curve.groupby(np.arange(len(curve)) // 5).agg({"nm": 'mean', 'value': 'mean'})
        curve[curve < 0] = 0
        extracted = pd.DataFrame(fft_for_curve(curve['value'].values))
        extracted = scaler.fit_transform(extracted).transpose()
        return extracted, curve

def transformTenNums(predicted_ten_nums, scaler):
    predicted_ten_nums = scaler.inverse_transform([predicted_ten_nums]).squeeze()
    predicted_ten_nums = [int(ele) for ele in predicted_ten_nums]
    predicted_ten_nums = [100 if ele > 100 else ele for ele in predicted_ten_nums]
    print(predicted_ten_nums)
    return predicted_ten_nums


def main():
    extracted,curve = readAndCurateCurve(input_curve_file, scalerX)
    fft_tensor = torch.FloatTensor(extracted)
    predicted_ten_nums = model(fft_tensor)
    predicted_ten_nums = transformTenNums(predicted_ten_nums.detach().numpy().squeeze(), scalerY)
    lsr.set_column_data(1, predicted_ten_nums)
    lsr.set_column_data(2, lsr.compute_column_based_on_first(0.7))
    lsr.set_column_data(3, lsr.compute_column_based_on_first(0.5))
    lsr.set_column_data(4, lsr.compute_column_based_on_first(0.3))
    lsr.run()  # this take 1 sec
    # Lets wait for the curve
    time.sleep(2)

    # Spectra has to point to example_database folder before starting
    save_curve("{}".format(sample_name + "_recreated.ssm"))
    print("Waiting for recreated file to be saved...")
    time.sleep(2)
    while not os.path.exists("example_database/{}".format(sample_name + "_recreated.ssm")):
        time.sleep(1)
    extracetd_reconstructed, reconstructed_curve = readAndCurateCurve("example_database/{}".format(sample_name + "_recreated.ssm"), scalerX)

    mse = ((reconstructed_curve.value - curve.value)**2).mean(axis=0)
    print(mse)
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(curve.nm, curve.value)
    plt.subplot(122)
    plt.plot(reconstructed_curve.nm, reconstructed_curve.value, 'r')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if not admin.isUserAdmin():
       admin.runAsAdmin()
    main()
