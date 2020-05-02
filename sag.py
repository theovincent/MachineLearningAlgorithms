from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.train_sag_classifier import train_sag_classifier, sag_classifier_test
from src.losses.hinge_loss import hinge_loss, hinge_derivative
from src.utils.preprocessing.read_file.read_csv import read_csv
from src.utils.preprocessing.standardise import standardize_data
from src.utils.preprocessing.normalisation import normalisation_data
from src.utils.preprocessing.pca import pca
from src.utils.preprocessing.split_label import split_label


# --- Get the data ---
CSV_PATH = Path("data/data.csv")
(HOUSES_TRAIN, PRICES_TRAIN, HOUSES_VALIDATION, PRICES_VALIDATION, HOUSES_TEST, PRICES_TEST) = read_csv(CSV_PATH)


# --- Transform the data ---
print("Get the data and transform it ...")
# Normalisation
NORM_TRAIN = normalisation_data(HOUSES_TRAIN)
NORM_VALID = normalisation_data(HOUSES_VALIDATION)
NORM_TEST = normalisation_data(HOUSES_TEST)

# Standardisation
STAN_TRAIN = standardize_data(HOUSES_TRAIN)
STAN_VALID = standardize_data(HOUSES_VALIDATION)
STAN_TEST = standardize_data(HOUSES_TEST)

# PCA
NB_CUTOFF = 5
CUTOFFS = np.linspace(0.50, 0.9, NB_CUTOFF)
PCAS_TRAIN = []
PCAS_VALID = []
PCAS_TEST = []
# With whitening
PCASW_TRAIN = []
PCASW_VALID = []
PCASW_TEST = []
for cutoff in CUTOFFS:
    PCAS_TRAIN.append(pca(HOUSES_TRAIN, cutoff))
    # All the sets have to have the same number of features
    NB_FEATURE = PCAS_TRAIN[-1].shape[1]
    PCAS_VALID.append(pca(HOUSES_VALIDATION, 1)[:, :NB_FEATURE])
    PCAS_TEST.append(pca(HOUSES_TEST, 1)[:, :NB_FEATURE])
    # With whitening
    PCASW_TRAIN.append(pca(HOUSES_TRAIN, cutoff, whitening=True))
    # All the sets have to have the same number of features
    NB_FEATUREW = PCASW_TRAIN[-1].shape[1]
    PCASW_VALID.append(pca(HOUSES_VALIDATION, 1, whitening=True)[:, :NB_FEATUREW])
    PCASW_TEST.append(pca(HOUSES_TEST, 1, whitening=True)[:, :NB_FEATUREW])


# Gather the data
ALL_TRAINS = [NORM_TRAIN, STAN_TRAIN]
ALL_TRAINS.extend(PCAS_TRAIN)
ALL_TRAINS.extend(PCASW_TRAIN)

ALL_VALIDS = [NORM_VALID, STAN_VALID]
ALL_VALIDS.extend(PCAS_VALID)
ALL_VALIDS.extend(PCASW_VALID)

ALL_TEST = [NORM_TEST, STAN_TEST]
ALL_TEST.extend(PCAS_TEST)
ALL_TEST.extend(PCASW_TEST)

LIST_PREPROCESS = ["Normalisation", "Standardization"] + ["PCA {}".format(cut) for cut in CUTOFFS]
LIST_PREPROCESS += ["PCA Whitening {}".format(cut) for cut in CUTOFFS]

# Split the labels
SPILT_TRAIN = split_label(PRICES_TRAIN, PRICES_TRAIN)
SPILT_VALID = split_label(PRICES_VALIDATION, PRICES_TRAIN)
SPILT_TEST = split_label(PRICES_TEST, PRICES_TRAIN)


# --- Train the SAG Classifiers ---
print("Train the SAG...")
NB_TRAININGS = len(ALL_TRAINS)
ACCURACIES = np.zeros(NB_TRAININGS)
ACCURACY_MAX = 0
LAMBDA_OPT = 0
ETA_OPT = 0
IDX_TRY_OPT = None
for idx_try in range(NB_TRAININGS):
    print(LIST_PREPROCESS[idx_try])
    # Training with the parameters
    RESULTS = train_sag_classifier(ALL_TRAINS[idx_try], SPILT_TRAIN, ALL_VALIDS[idx_try], SPILT_VALID, hinge_loss,
                                   hinge_derivative)
    (ACCURACY_VALID, LAMBDA, ETA) = RESULTS

    # Update the global parameters
    ACCURACIES[idx_try] = ACCURACY_VALID
    print("Validation accuracy", ACCURACY_VALID)
    if ACCURACY_MAX < ACCURACY_VALID:
        ACCURACY_MAX = ACCURACY_VALID
        LAMBDA_OPT = LAMBDA
        ETA_OPT = ETA
        IDX_TRY_OPT = idx_try


# --- Testing with the best parameters ---
print("Test the SAG...")
ACCURACY_TEST = sag_classifier_test(ALL_TRAINS[IDX_TRY_OPT], SPILT_TRAIN, ALL_VALIDS[IDX_TRY_OPT], SPILT_VALID,
                                    hinge_loss, hinge_derivative, ETA_OPT, LAMBDA_OPT)

print("The accuracy for the test set is :", ACCURACY_TEST)
print("It was made with the preprocessing :", LIST_PREPROCESS[IDX_TRY_OPT])
print("The optimal value of lambda is :", LAMBDA_OPT)
print("The optimal value of eta is :", ETA_OPT)

# Plot the losses
plt.plot(np.arange(0, NB_TRAININGS, 1), ACCURACIES)
plt.xlabel("Different preprocessing")
plt.ylabel("Validation accuracy")
plt.show()

