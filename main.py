import random

from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from AutoEncoder import AutoEncoder
from DataContrainer import  Data
from LSR_comm import LSR_comm
import torch
import time
import numpy as np
from matplotlib import pyplot as plt

from Predict10 import Predict10

STOP_THRESHOLD = 1e-2

def plot_curve(df, new_curve):
    a = plt.scatter(df['nm'], df['value'])
    b = plt.scatter(df['nm'], new_curve)
    plt.legend((a, b), ('Ref Curve', 'New Curve'))
    plt.show()

def main(lsr, model_param, optimizer_param, loss_function_param, encoded_param, cnt=0):
    print("Round: ", cnt)
    cnt += 1

    lsr.set_column_data(1, encoded_param)
    lsr.set_column_data(2, lsr.compute_column_based_on_first(0.7))
    lsr.set_column_data(3, lsr.compute_column_based_on_first(0.5))
    lsr.set_column_data(4, lsr.compute_column_based_on_first(0.3))
    lsr.run()
    time.sleep(5) # at this time we should open spectra wiz and save the file

    print("\t Reading new HyperOCR data...")
    # Read HYperOCR (Current Curve)
    ocr = Data("/home/mateo/LSR/ZAGREB071022/Akrozprozor.ssm")
    sensor_reading = ocr.randomize_the_data_a_bit(ocr.get_data())
    #sensor_reading = ocr.get_data()

    # Read RefData
    cm = Data("/home/mateo/LSR/ZAGREB071022/Bsoba.ssm")
    ref = cm.get_data()


    # Find 10 numbers that match those curves the best
    model, optimizer, loss_function, new_10, loss = find_params(model_param=model_param,
                                                                 optimizer_param=optimizer_param,
                                                                 loss_function_param=loss_function_param,
                                                                 ref_param=ref,
                                                                 current_curve=sensor_reading,
                                                                 encoded_param=encoded_param)
    if loss >= STOP_THRESHOLD:
        main(lsr=lsr, model_param=model, optimizer_param=optimizer,
             loss_function_param=loss_function, encoded_param=new_10, cnt=cnt)
    else:
        #lsr.stop()
        print("Curve found...")
        print(new_10)
        print("Stopping..")

def generate_random():
    max = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    return random.sample(range(1, 100), 10)

def min_max_transform(X,in_min=0, in_max=6, out_min=0, out_max=100):
    old_range = (in_max - in_min)
    new_range = (out_max - out_min)
    new_val = (((X - in_min) * new_range) / old_range) + out_min
    return new_val

def find_params(model_param, optimizer_param, loss_function_param, ref_param, current_curve, encoded_param):

    ref = Variable(torch.Tensor(ref_param['value'].values),requires_grad=True)
    current_curve = Variable(torch.Tensor(current_curve['value'].values), requires_grad=True)
    encoded_param = torch.Tensor([encoded_param])
    input_to_model = encoded_param
    for i in range(100):
        input_to_model = model_param(input_to_model)
        loss = loss_function_param(current_curve, ref)
        optimizer_param.zero_grad()
        loss.backward()
        optimizer_param.step()

    new_10 = input_to_model.squeeze().detach().cpu()
    plot_curve(ref_param, current_curve.detach().cpu())
    new_10 = [int(min_max_transform(item))  for item in new_10]
    #print("\t Current 10 vals: ", encoded)
    print("\t Average Error between curves: ", np.mean(loss.item()))

    return model_param, optimizer_param, loss_function_param, new_10,  np.mean(loss.item())


if __name__ == "__main__":
    # /dev/cu.usbmodem142201, COM3,/dev/ttyACM0
    lsr = LSR_comm("/dev/ttyACM0")
    time.sleep(1)
    model = Predict10()
    model.to("cpu")
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-2)
    main(lsr=lsr, model_param=model, optimizer_param=optimizer, loss_function_param=loss_function, encoded_param=generate_random(), cnt=0)
