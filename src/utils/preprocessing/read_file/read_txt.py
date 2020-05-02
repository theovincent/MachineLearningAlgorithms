"""
This module has the purpose of reading a txt file.
"""
from pathlib import Path
import numpy as np


def read_txt(name_path, training_ratio=0.8):
    # Get the data
    data = []
    labels = []
    with open(name_path, 'r') as file:
        for line in file.readlines():
            line = line.rstrip()
            line = line.split(",")
            data.append([np.float(line[0]), np.float(line[1])])
            labels.append(np.float(line[-1]))

    # Convert the lists to arrays
    data = np.array(data)
    labels = np.array(labels)

    # Split the data into two groups : idx_try set and validation set
    nb_points = len(data)
    data_train = data[: int(nb_points * training_ratio)]
    labels_train = labels[: int(nb_points * training_ratio)]
    data_validation = data[int(nb_points * training_ratio):]
    labels_validation = labels[int(nb_points * training_ratio):]

    return data_train, labels_train, data_validation, labels_validation


if __name__ == "__main__":
    DATA_PATH = Path("../../../data/normal_data.txt")
    (DATA_TRAIN, LABEL_TRAIN, DATA_VALID, LABEL_VALID) = read_txt(DATA_PATH)
    print("Data \n", len(DATA_TRAIN))
    print("Label \n", len(LABEL_TRAIN))
