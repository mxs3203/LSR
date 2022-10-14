
import csv
import random

import pandas as pd

class HyperOCR:

    def __init__(self, file_to_read, data_range=[350, 850]):
        self.file_to_read = file_to_read
        self.data_frame = self.read_new_data(file_to_read)
        self.data_frame = self.data_frame.loc[(self.data_frame['nm'] >= data_range[0]) & (self.data_frame['nm'] <= data_range[1])]
        self.info = ""

    def read_new_data(self, file_to_read):
        cols = ['nm', 'value']
        with open(file_to_read) as f:
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

    def randomize_the_data_a_bit(self):
        change = self.data_frame.sample(200).index
        self.data_frame.loc[change, 'value'] = self.data_frame.loc[change, 'value'] + random.randint(0, 100)
        return self.data_frame


# ocr = HyperOCR("1.ssm")
# print(ocr.get_data())
# ocr.randomize_the_data_a_bit()
# print(ocr.get_data())