import csv
import pandas as pd

class RefData:

    def __init__(self, path_to_ref,  data_range=[350, 850]):
        self.path_to_ref = path_to_ref
        self.ref_data = self.read_new_data(path_to_ref)
        self.ref_data = self.ref_data.loc[(self.ref_data['nm'] >= data_range[0]) & (self.ref_data['nm'] <= data_range[1])]
        self.info = ""


    def read_new_data(self, file_to_read):
        cols = ['nm', 'value']
        with open(file_to_read) as f:
            data = csv.reader(f, delimiter="\t")
            cnt = 0
            lst = []
            for d in data: # For every line
                if not cnt in [0]: # skip first line
                    parts = d[0].split(" ") # split string by space
                    if len(parts) == 4:
                        lst.append([parts[1], parts[3]])

                else:
                    self.info = d[0]
                    #print("We skip first line: ", d)
                cnt += 1
            data_frame = pd.DataFrame(lst, columns=cols, dtype='float64')
            f.close()
        return data_frame

    def get_info(self):
        return self.info

    def get_data(self):
        return self.ref_data


# cm = RefData("/Users/au589901/PycharmProjects/LSR_commands/ZAGREB071022/MORE10cm/more.IRR")
# print(cm.get_data())