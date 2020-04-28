"""
This module has the purpose of reading a txt file.
"""
from pathlib import Path
import numpy as np


def read_txt(name_path):
    data = []
    label = []
    with open(name_path, 'r') as file:
        for line in file.readlines():
            line = line.rstrip()
            line = line.split(",")
            data.append([np.float(line[0]), np.float(line[1])])
            label.append(np.float(line[-1]))

    return np.array(data), np.array(label)


if __name__ == "__main__":
    DATA_PATH = Path("../../../data/small_data.txt")
    (DATA, LABEL) = read_txt(DATA_PATH)
    print("Data \n", DATA)
    print("Label \n", LABEL)