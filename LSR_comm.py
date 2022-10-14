import time

import serial


class LSR_comm:

    def __init__(self, com_port):
        self.S = serial.Serial(com_port)
        self.S.baudrate = 1000000
        self.S.bytesize = 8
        self.S.parity = 'N'
        self.S.stopbits = 1
        self.columns_with_data = []
        self.column_1 = []

    def send_any_command(self, msg):
        self.S.write(bytes(msg, 'utf-8'))
        time.sleep(0.05)
        response = self.S.readline()
        print("\t LSR reponsed: ", response)

    def ask_for_status(self):
        msg = "{\"READ\": \"status\"}"
        self.send_any_command(msg)

    def set_column_data(self, column, list_of_nums):
        if len(list_of_nums) == 10:
            if column == 1:
                self.column_1 = list_of_nums
            msg = "{" + "\"Col-{}\": [{},{},{},{},{},{},{},{},{},{}]".format(column,list_of_nums[0],list_of_nums[1],list_of_nums[2],
                                                                     list_of_nums[3],list_of_nums[4], list_of_nums[5],
                                                                     list_of_nums[6],list_of_nums[7],list_of_nums[8],
                                                                     list_of_nums[9]) + "}"
            self.send_any_command(msg)
            self.columns_with_data.append(column)
        else:
            print("\t List should contain exactly 10 numbers")

    # Generate second,third or fourth column based on values of first column (75%,50% and 30% intesity)
    def compute_column_based_on_first(self, coef):
        col_vals = []
        if 1 in self.columns_with_data:
            for i in self.column_1:
                col_vals.append(int(i * coef))

        return col_vals

    def set_block_temp(self, temp):
        msg  = "{" + "\"Tblock\": {}".format(temp) + "}"
        self.send_any_command(msg)

    def run(self):
        if 1 in self.columns_with_data and 2 in self.columns_with_data and 3 in self.columns_with_data and 4 in self.columns_with_data:
            msg = "{\"DO\": \"run\"}"
            self.send_any_command(msg)
            self.columns_with_data = []
        else:
            print("\t ERROR: All column values should be set first")

    def stop(self):
        msg = "{\"DO\": \"stop\"}"
        self.send_any_command(msg)



# LSR = LSR_comm("/dev/cu.usbmodem142201")
# LSR.ask_for_status()
# LSR.set_column_data(1, [12,3,2,2,4,2,4,9,23,99])
# LSR.set_column_data(4, [2,30,22,21,44,22,44,1,3,29])
# LSR.set_block_temp(22)
# LSR.run()
# LSR.set_column_data(3, [2,30,22,21,44,22,44,1,3,29])
# LSR.set_column_data(2, [2,30,22,21,44,22,44,1,3,29])
# LSR.run()
# time.sleep(2)
# LSR.stop()