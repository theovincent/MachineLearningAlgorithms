from pathlib import Path
import numpy as np
from src.utils.read_file.read_txt import read_txt


def standardize_data(data):
    """
    Standardizes the data. Set the mean to 0 and the standard deviation to 1
    for each feature

    Args:
        data (array): the data to standardize.

    Return:
        standardized_data (array): the standardizes data.
    """
    return (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 10 ** -16)


if __name__ == "__main__":
    DATA_PATH = Path("../../data/small_data.txt")
    (DATA_TEST, LABEL) = read_txt(DATA_PATH)
    STAND_DATA = standardize_data(DATA_TEST)
    print("Standardize data \n", STAND_DATA)
    print("Mean first feature", np.mean(STAND_DATA[:, 0]))
    print("Standard deviation first feature", np.std(STAND_DATA[:, 0]))