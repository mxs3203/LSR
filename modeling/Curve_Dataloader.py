from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler



class Curve_Loader(Dataset):

    def __init__(self, csv_file,fft_size):

        self.annotation = pd.read_csv(csv_file, sep=",", )
        self.annotation = self.annotation.iloc[:, 1:] # remove first column
        self.scalerX = MinMaxScaler()
        self.scalerX_fft = MinMaxScaler()
        self.scalerY = MinMaxScaler()
        self.y = pd.DataFrame(self.scalerY.fit_transform(self.annotation.iloc[:, :10])) # first 10 nums
        self.x_fft = pd.DataFrame(self.scalerX.fit_transform(self.annotation.iloc[:, 11:(11+fft_size)])) # raw curve
        self.x = pd.DataFrame(self.scalerX_fft.fit_transform(self.annotation.iloc[:, (11+fft_size)::]))  # fft values
        self.x_both = pd.DataFrame(self.scalerX_fft.fit_transform(self.annotation.iloc[:, 11::]))  # fft values

    def getScalerX(self):
        return self.scalerX_fft

    def getScalerY(self):
        return self.scalerY

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        y = np.array(self.y.iloc[idx, :], dtype="float")
        x = np.array(self.x_fft.iloc[idx, :], dtype="float")

        return x, y