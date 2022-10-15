import random

from torch.utils.data import TensorDataset, DataLoader

from AutoEncoder import AutoEncoder
from DataContrainer import  Data
from LSR_comm import LSR_comm
import torch
import time
import numpy as np
from matplotlib import pyplot as plt

STOP_THRESHOLD = 1e-2

def plot_curve(df, new_curve):
    a = plt.scatter(df['nm'], df['value'])
    b = plt.scatter(df['nm'], new_curve)
    plt.legend((a, b), ('Ref Curve', 'New Curve'))
    plt.show()

def main(lsr, model_param, optimizer_param, loss_function_param, encoded_param, cnt=0):
    print("Round: ", cnt)
    cnt += 1

    print("\t Reading new HyperOCR data...")
    # Read HYperOCR (Current Curve)
    ocr = Data("/home/mateo/LSR/ZAGREB071022/Akrozprozor.ssm")
    sensor_reading = ocr.randomize_the_data_a_bit(ocr.get_data())
    #sensor_reading = ocr.get_data()

    # Read RefData
    cm = Data("/home/mateo/LSR/ZAGREB071022/Bsoba.ssm")
    ref = cm.get_data()

    lsr.set_column_data(1, encoded_param)
    lsr.set_column_data(2, lsr.compute_column_based_on_first(0.7))
    lsr.set_column_data(3, lsr.compute_column_based_on_first(0.5))
    lsr.set_column_data(4, lsr.compute_column_based_on_first(0.3))
    lsr.run()

    # Find 10 numbers that match those curves the best
    model, optimizer, loss_function, encoded, loss = find_params(model_param, optimizer_param, loss_function_param, ref, sensor_reading)
    if loss >= STOP_THRESHOLD:
        main(lsr, model, optimizer, loss_function, encoded, cnt)
    else:
        #lsr.stop()
        print("Curve found...")
        print(encoded)
        print("Stopping..")

def generate_random():
    max = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    return random.sample(range(1, 100), 10)

def min_max_transform(X,in_min=0, in_max=600, out_min=0, out_max=100):
    old_range = (in_max - in_min)
    new_range = (out_max - out_min)
    new_val = (((X - in_min) * new_range) / old_range) + out_min
    return new_val

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
    plot_curve(ref_param, reconstructed.detach().cpu())
    encoded = [int(item *100)  for item in encoded]
    print("\t Current 10 vals: ", encoded)
    print("\t Average Error between curves: ", np.mean(losses))

    return model_param, optimizer_param, loss_function_param, encoded, np.mean(losses)


if __name__ == "__main__":
    # /dev/cu.usbmodem142201, COM3,/dev/ttyACM0
    lsr = LSR_comm("/dev/ttyACM0")
    time.sleep(1)
    model = AutoEncoder(input_size=1001)
    model.to("cpu")
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    main(lsr, model, optimizer, loss_function, generate_random(), cnt=0)
