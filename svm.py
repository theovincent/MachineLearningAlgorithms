from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.train_sag_classifier import train_sag_classifier, sag_classifier_test
from src.losses.hinge_loss import hinge_loss, hinge_derivative
from src.utils.preprocess import get_data
from src.utils.preprocessing.split_label import split_label


# --- Get the data ---
CSV_PATH = Path("data/data.csv")
(ALL_TRAINS, ALL_VALIDS, ALL_TESTS, PRICES_TRAIN, PRICES_VALID, PRICES_TEST, LIST_PREPROCESS) = get_data(CSV_PATH)

# Split the labels
SPLIT_TRAIN = split_label(PRICES_TRAIN, PRICES_TRAIN)
SPLIT_VALID = split_label(PRICES_VALID, PRICES_TRAIN)
SPLIT_TEST = split_label(PRICES_TEST, PRICES_TRAIN)

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
    RESULTS = train_sag_classifier(ALL_TRAINS[idx_try], SPLIT_TRAIN, ALL_VALIDS[idx_try], SPLIT_VALID, hinge_loss,
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
ACCURACY_TEST = sag_classifier_test(ALL_TRAINS[IDX_TRY_OPT], SPLIT_TRAIN, ALL_TESTS[IDX_TRY_OPT], SPLIT_VALID,
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
