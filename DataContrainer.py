
import csv
import random

import pandas as pd

class Data:

    def __init__(self, file_to_read, data_range=[350, 850]):
        self.file_to_read = file_to_read
        self.data_frame = self.read_new_data()
        self.data_frame = self.data_frame.loc[(self.data_frame['nm'] >= data_range[0]) & (self.data_frame['nm'] <= data_range[1])]
        self.info = ""
        self.min_nm = data_range[0]
        self.max_nm = data_range[1]
        self.data_frame = self.data_frame.iloc[::5, :]

    def read_new_data(self):
        cols = ['nm', 'value']
        with open(self.file_to_read) as f:
            data = csv.reader(f, delimiter="\t")
            cnt = 0
            lst = []
            for d in data:
                if cnt != 0:
                    parts = d[0].split(" ")
                    if len(parts) == 4:
                        lst.append([parts[1], parts[3]])

                else:
                    self.info = d[0]
                    #print("We skip first line: ", d)
                cnt += 1
            data_frame = pd.DataFrame(lst, columns=cols, dtype='float64')
            f.close()
        return data_frame

    def get_data(self):
        return self.data_frame

    def get_info(self):
        return self.info

    def randomize_the_data_a_bit(self, df):
        change = df.sample(100).index
        df.loc[change, 'value'] = df.loc[change, 'value'] + random.randint(-30000, 30000)
        return df


# ocr = HyperOCR("1.ssm")
# print(ocr.get_data())
# ocr.randomize_the_data_a_bit()
# print(ocr.get_data())