import os
import random

from torch.utils.data import TensorDataset, DataLoader

import admin
from DataContrainer import  Data
from LSR_comm import LSR_comm
import torch
import pandas as pd
import time
import numpy as np
from matplotlib import pyplot as plt

from SpectraWizSaver import save_curve
from modeling.AutoEncoder import AutoEncoder

STOP_THRESHOLD = 0.001
EPS = 0.0001

def readAndCurateCurve(file):
    with open(file, 'rb') as f2:
        curve = pd.read_csv(f2, delimiter=" ", skiprows=1, names=['nm', 'ignore', 'value'])
        curve = curve.loc[(curve['nm'] >= 350) & (curve['nm'] <= 850)]
        curve = curve.groupby(np.arange(len(curve)) // 5).agg({"nm": 'mean', 'value': 'mean'})
        curve[curve < 0] = 0
        curve['value'] = transformToLog10(curve['value'] + EPS)
        return curve

def transformToLog10(x):
    return np.log10(x)

def transfromFromLog10(x):
    return 10**x

def plot_curve(df, new_curve):
    a = plt.scatter(df['nm'], df['value'])
    b = plt.scatter(df['nm'], new_curve)
    plt.legend((a, b), ('Ref Curve', 'New Curve'))
    plt.show()

def main(lsr, model_param, optimizer_param, loss_function_param, encoded_param, ref_df, cnt=0):
    print("Round: ", cnt)
    # Start LSR with params
    lsr.set_column_data(1, encoded_param)
    lsr.set_column_data(2, lsr.compute_column_based_on_first(0.7))
    lsr.set_column_data(3, lsr.compute_column_based_on_first(0.5))
    lsr.set_column_data(4, lsr.compute_column_based_on_first(0.3))
    lsr.run()

    # Spectra has to point to example_database folder before starting
    save_curve("{}".format("recreated.ssm"))
    print("Waiting for recreated file to be saved...")
    time.sleep(2)
    while not os.path.exists("example_database/{}".format("recreated.ssm")):
        time.sleep(1)

    print("\t Reading new HyperOCR data...")
    # Read HYperOCR (Current Curve)
    sensor_reading = readAndCurateCurve("example_database/recreated.ssm")

    cnt += 1
    # Find 10 numbers that match those curves the best
    model, optimizer, loss_function, encoded, loss = find_params(model_param, optimizer_param, loss_function_param, ref_df, sensor_reading)
    if loss >= STOP_THRESHOLD:
        main(lsr, model, optimizer, loss_function, encoded, ref_df, cnt)
    else:
        #lsr.stop()
        print("Curve found...")
        print(encoded)
        print("Stopping..")

def generate_random():
    max = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    return np.array(random.sample(range(1, 1000), 10), dtype="int")

def find_params(model_param, optimizer_param, loss_function_param, ref_param, current_curve_param):
    current_curve = torch.Tensor([current_curve_param['value'].values])
    ref = torch.Tensor([ref_param['value'].values])
    my_dataset = TensorDataset(current_curve, ref)
    my_dataloader = DataLoader(my_dataset, batch_size=1)
    losses = []

    t_end = time.time() + 1
    print("\t Optimizing for 1 secs")
    while(time.time() <= t_end):
        for x, y in my_dataloader:
            reconstructed, encoded = model_param(x)
            loss = loss_function_param(reconstructed, y)
            optimizer_param.zero_grad()
            loss.backward()
            optimizer_param.step()
            losses.append(loss.item())

    encoded = encoded.squeeze().tolist()
    encoded = int((encoded * 1000.0)) # model returns sigmoid
    plot_curve(ref_param, reconstructed.detach().cpu())
    print("\t Current 10 vals: ", encoded)
    print("\t Average Error between curves: ", np.mean(losses))

    return model_param, optimizer_param, loss_function_param, encoded, np.mean(losses)


if __name__ == "__main__":
    if not admin.isUserAdmin():
       admin.runAsAdmin()

    # /dev/cu.usbmodem142201, COM3,/dev/ttyACM0
    # Read RefData
    ref = readAndCurateCurve("ZAGREB071022/Akrozprozor.ssm")

    lsr = LSR_comm("COM3")
    time.sleep(1)
    model = AutoEncoder(input_size=201)
    model.to("cpu")
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)

    main(lsr, model, optimizer, loss_function, generate_random(), ref, cnt=0)