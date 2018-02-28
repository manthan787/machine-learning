import numpy as np

DATA_PATH = "data/"


def read_data(name="data_1_small"):
    with open(DATA_PATH + name + ".txt") as f:
        return np.loadtxt(f, delimiter=" ", skiprows=0)