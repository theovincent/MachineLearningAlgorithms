from pathlib import Path
import numpy as np
from src.utils.preprocessing.read_file.read_csv import read_csv
from src.utils.preprocessing.standardise import standardize_data
from src.utils.preprocessing.normalisation import normalisation_data
from src.utils.preprocessing.pca import pca
from src.utils.preprocessing.split_label import split_label


def get_houses_data(csv_path):
    (houses_train, prices_train, houses_validation, prices_validation, houses_test, prices_test) = read_csv(csv_path)

    # Normalisation
    norm_train = normalisation_data(houses_train)
    norm_valid = normalisation_data(houses_validation)
    norm_test = normalisation_data(houses_test)

    # Standardisation
    stan_train = standardize_data(houses_train)
    stan_valid = standardize_data(houses_validation)
    stan_test = standardize_data(houses_test)

    # PCA
    nb_cutoff = 5
    cutoffs = np.linspace(0.50, 0.9, nb_cutoff)
    pcas_train = []
    pcas_valid = []
    pcas_test = []
    # With whitening
    pcasw_train = []
    pcasw_valid = []
    pcasw_test = []
    for cutoff in cutoffs:
        pcas_train.append(pca(houses_train, cutoff))
        # All the sets have to have the same number of features
        nb_feature = pcas_train[-1].shape[1]
        pcas_valid.append(pca(houses_validation, 1)[:, :nb_feature])
        pcas_test.append(pca(houses_test, 1)[:, :nb_feature])
        # With whitening
        pcasw_train.append(pca(houses_train, cutoff, whitening=True))
        # All the sets have to have the same number of features
        nb_featurew = pcasw_train[-1].shape[1]
        pcasw_valid.append(pca(houses_validation, 1, whitening=True)[:, :nb_featurew])
        pcasw_test.append(pca(houses_test, 1, whitening=True)[:, :nb_featurew])


    # Gather the data
    all_trains = [norm_train, stan_train]
    all_trains.extend(pcas_train)
    all_trains.extend(pcasw_train)

    all_valids = [norm_valid, stan_valid]
    all_valids.extend(pcas_valid)
    all_valids.extend(pcasw_valid)

    all_tests = [norm_test, stan_test]
    all_tests.extend(pcas_test)
    all_tests.extend(pcasw_test)

    list_preprocess = ["Normalisation", "Standardization"] + ["PCA {}".format(cut) for cut in cutoffs]
    list_preprocess += ["PCA Whitening {}".format(cut) for cut in cutoffs]

    return all_trains, all_valids, all_tests, prices_train, prices_validation, prices_test, list_preprocess


if __name__ == "__main__":
    # Define the path
    CSV_PATH = Path("../../data/data.csv")

    # --- Get the data ---
    (ALL_TRAIN, ALL_VALID, ALL_TEST, SPLIT_TRAIN, SPLIT_VALID, SPILT_TEST, LIST_PREPROCESS) = get_houses_data(CSV_PATH)
