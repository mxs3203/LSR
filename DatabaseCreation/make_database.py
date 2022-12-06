import os
import pickle
import random

from DatabaseCreation import admin
from TrainingDataCreation.DataContrainer import  Data
from TrainingDataCreation.DatabaseItem import Item
from LSR_comm import LSR_comm
import time

from DatabaseCreation.SpectraWizSaver import save_curve

DIT = 10

def main():

    cnt = 793
    # /dev/cu.usbmodem142201, COM3,/dev/ttyACM0
    lsr = LSR_comm("COM3")
    time.sleep(1) # waiting for autsomation start

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
        item = Item(curve=ocr, ten_nums=ten_nums,
                    file_name="example_database/{}.ssm".format(cnt),
                    cnt=cnt,
                    DIT=DIT)
        with open(r'example_database/train_data/obs_{}.pickle'.format(cnt), 'wb') as f:
            pickle.dump(item, f)
            f.close()

        with open(r'example_database/train_data/obs_{}.pickle'.format(cnt), 'rb') as f:
            what_is_here = pickle.load(f)
            f.close()
        assert (what_is_here.curve.data_frame.empty) == False
        cnt = cnt + 1



def generate_random(): # [70,50,30,80,100,80, 50, 50,80,80]
    # Realistic Curves setup
    nums = [random.sample(range(10, 90), 1),
            random.sample(range(10, 70), 1),
            random.sample(range(10, 80), 1),
            random.sample(range(50, 100), 1),
            random.sample(range(50, 100), 1),
            random.sample(range(40, 80), 1),
            random.sample(range(40, 90), 1),
            random.sample(range(30, 90), 1),
            random.sample(range(20, 90), 1),
            random.sample(range(20, 90), 1)]
    return [element for nestedlist in nums for element in nestedlist]
    # single channel setup
    #nums = [[0],[0],[0],[0],[0],[0],[0],[0], [0],random.sample(range(0, 100), 1)]
    #return [element for nestedlist in nums for element in nestedlist]
    # complete randomness
    #return random.sample(range(0, 100), 10)


if __name__ == "__main__":
    if not admin.isUserAdmin():
       admin.runAsAdmin()
    main()