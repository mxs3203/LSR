
import glob
import pickle

with open("/home/mateo/Desktop/LSR_data/train_data/obs_4.pickle", 'a') as f:
    a = pickle.load(f)

for f in glob.glob("/home/mateo/Desktop/LSR_data/train_data/*.pickle"):
    with open(f, 'rb') as file:
        pickle_file = pickle.load(file)
        print()