from pathlib import Path
import numpy as np
from src.utils.preprocessing.read_file.read_txt import read_txt


def normalisation_data(data):
    """
    Standardizes the data. Set the mean to 0 and the standard deviation to 1
    for each feature

    Args:
        data (array): the data to standardize.

    Return:
        standardized_data (array): the standardizes data.
    """
    return data / (np.linalg.norm(data, axis=0) + 10 ** -16)


if __name__ == "__main__":
    DATA_PATH = Path("../../../data/normal_data.txt")
    (DATA_TRAIN, LABEL_TRAIN, DATA_VALID, LABEL_VALID) = read_txt(DATA_PATH)
    STAND_DATA = normalisation_data(DATA_TRAIN)
    print("Number standardized data", len(STAND_DATA))
    print("The norm of each feature", np.linalg.norm(STAND_DATA, axis=0))
