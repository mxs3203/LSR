from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler



class Curve_Loader(Dataset):

    def __init__(self, csv_file,fft_size=11):

        self.annotation = pd.read_csv(csv_file, sep=",", )
        self.annotation = self.annotation.iloc[:, 1:] # remove first column
        self.y = pd.DataFrame(self.annotation.iloc[:, :10]) # first 10 nums
        #self.x = pd.DataFrame(self.annotation.iloc[:, 11:(11+fft_size)]) # fft numbers
        self.x = pd.DataFrame(self.annotation.iloc[:, (11+fft_size+1)::])  # curve
        print("Curve",self.x.describe())
        print("Ten Nums",self.y.describe())

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        y = np.array(self.y.iloc[idx, :], dtype="float")
        x = np.array(self.x.iloc[idx, :], dtype="float")

        return x, y