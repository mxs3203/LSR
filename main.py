import random

from torch.utils.data import TensorDataset, DataLoader

from AutoEncoder import AutoEncoder, WeightClipper
from HyperOCR import HyperOCR
from LSR_comm import LSR_comm
import torch
import time
import timeit
import numpy as np

from RefData import RefData

TIME_FOR_HYPEROCR_READ = 0.2  # SEC
STOP_THRESHOLD = 1e-3


def main(model_param, optimizer_param, loss_function_param, encoded_param, cnt=0):
    print("Round: ", cnt)
    cnt += 1

    lsr = LSR_comm("/dev/cu.usbmodem142201")
    lsr.ask_for_status()
    lsr.set_column_data(1, encoded_param)
    lsr.set_column_data(2, lsr.compute_column_based_on_first(0.75))
    lsr.set_column_data(3, lsr.compute_column_based_on_first(0.5))
    lsr.set_column_data(4, lsr.compute_column_based_on_first(0.3))
    lsr.run()

    print("\t Reading new HyperOCR data...")
    time.sleep(TIME_FOR_HYPEROCR_READ)

    # Read HYperOCR (Current Curve)
    ocr = HyperOCR("1.ssm")
    sensor_reading = ocr.randomize_the_data_a_bit()


    # Read RefData
    cm = RefData("/Users/au589901/PycharmProjects/LSR_commands/ZAGREB071022/MORE10cm/more.IRR")
    ref = cm.get_data()

    # Find 10 numbers that match those curves the best
    model, optimizer, loss_function, encoded, loss = find_params(model_param, optimizer_param, loss_function_param, ref,
                                                           sensor_reading)
    if loss >= STOP_THRESHOLD:
        main(model, optimizer, loss_function, encoded, cnt)
    else:
        print("Curve found...")
        print(encoded)
        print("Stopping..")

def generate_random():
    return random.sample(range(1, 100), 10)


def find_params(model_param, optimizer_param, loss_function_param, ref_param, current_curve_param):
    current_curve = torch.Tensor([current_curve_param['value'].values])
    ref = torch.Tensor([ref_param['value'].values])
    my_dataset = TensorDataset(current_curve, ref)
    my_dataloader = DataLoader(my_dataset, batch_size=1)
    losses = []

    t_end = time.time() + 5
    print("\t Optimizing for 5 secs")
    while(time.time() <= t_end):
        for x, y in my_dataloader:
            reconstructed, encoded = model_param(x)
            loss = loss_function_param(reconstructed, y)
            optimizer_param.zero_grad()
            loss.backward()
            optimizer_param.step()
            losses.append(loss.item())

    encoded = encoded.squeeze().tolist()
    encoded = [int(item * 100) for item in encoded]
    print("\t Current 10 vals: ", encoded)
    print("\t Average Error between curves: ", np.mean(losses))

    return model_param, optimizer_param, loss_function_param, encoded, np.mean(losses)


if __name__ == "__main__":
    model = AutoEncoder(input_size=1001)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    main(model, optimizer, loss_function, generate_random(), cnt=0)
