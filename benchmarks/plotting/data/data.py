import pandas as pd


class data:
    def parse_data(path):
        data = pd.read_csv(path)
        return data

    def get_test_data():
        data = pd.read_csv('test_data.txt')
        return data

    def effective_bandwidth(data_transfered, time, rw: bool):
        if rw:
            return data_transfered / time * 2
        return data_transfered / time

    def flops(total_flops, time):
        return total_flops / time
