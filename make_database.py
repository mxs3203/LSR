import os
import pickle
import random

import admin
from DataContrainer import  Data
from DatabaseItem import Item
from LSR_comm import LSR_comm
import time
import numpy as np

from SpectraWizSaver import save_curve


def main():

    cnt = 1
    # /dev/cu.usbmodem142201, COM3,/dev/ttyACM0
    lsr = LSR_comm("COM3")
    time.sleep(1) # waiting for automation start

    while(True):
        print("Starting round: ", cnt)
        ten_nums = generate_random()
        print(ten_nums)
        lsr.set_column_data(1, ten_nums)
        lsr.set_column_data(2, lsr.compute_column_based_on_first(0.7))
        lsr.set_column_data(3, lsr.compute_column_based_on_first(0.5))
        lsr.set_column_data(4, lsr.compute_column_based_on_first(0.3))
        lsr.run()  # this take 1 sec
        # Lets wait for the curve
        time.sleep(1)
        # saving takes 0.3sec
        save_curve("{}".format(cnt))

        print("Waiting for file...", cnt,".ssm")
        while not os.path.exists("example_database/{}.ssm".format(cnt)):
            time.sleep(1)

        print("Reading Curve..")
        ocr = Data("example_database/{}.ssm".format(cnt))
        print("Saving pickle..")
        item = Item(curve=ocr, ten_nums=ten_nums, file_name="example_database/{}.ssm".format(cnt), cnt=cnt)
        with open('example_database/train_data/obs_{}.pickle'.format(cnt), 'wb') as handle:
            pickle.dump(item, handle, protocol=pickle.HIGHEST_PROTOCOL)
        cnt = cnt + 1



def generate_random():
    # nums = [random.sample(range(0, 40), 1),
    #         random.sample(range(0, 100), 1),
    #         random.sample(range(0, 100), 1),
    #         random.sample(range(0, 100), 1),
    #         random.sample(range(0, 100), 1),
    #         random.sample(range(0, 100), 1),
    #         random.sample(range(0, 100), 1),
    #         random.sample(range(0, 100), 1),
    #         random.sample(range(0, 100), 1),
    #         random.sample(range(0, 40), 1)]
    #return [element for nestedlist in nums for element in nestedlist]
    return random.sample(range(0, 100), 10)


if __name__ == "__main__":
    # if not admin.isUserAdmin():
    #     admin.runAsAdmin()
    main()