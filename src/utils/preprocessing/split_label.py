import numpy as np
from pathlib import Path
import numpy as np
import numpy.linalg as alg
from src.utils.preprocessing.read_file.read_txt import read_txt
from src.utils.preprocessing.read_file.read_csv import read_csv
from src.utils.preprocessing.standardise import standardize_data


def split_label(labels, threshold_labels):
    median_label = np.median(threshold_labels)
    nb_labels = len(labels)
    split_labels = np.ones(nb_labels) * -1

    for index_house in range(nb_labels):
        if labels[index_house] > median_label:
            split_labels[index_house] = 1

    return split_labels


if __name__ == "__main__":
    CSV_PATH = Path("../../../data/data.csv")

    # For csv
    (HOUSES_TRAIN, PRICES_TRAIN, HOUSES_VALIDATION, PRICES_VALIDATION, HOUSES_TEST, PRICES_TEST) = read_csv(CSV_PATH)
    SPILT_LABEL = split_label(PRICES_VALIDATION, PRICES_TRAIN)

    print("Split label", SPILT_LABEL)
