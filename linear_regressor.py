from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.train_sag_regressor import train_sag_regressor, sag_regressor_test
from src.losses.square_loss import square_loss, square_derivative, square_derivative_bias
from src.utils.preprocess import get_data


# --- Get the data ---
CSV_PATH = Path("data/data.csv")
(ALL_TRAINS, ALL_VALIDS, ALL_TESTS, PRICES_TRAIN, PRICES_VALID, PRICES_TEST, LIST_PREPROCESS) = get_data(CSV_PATH)


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
    RESULTS = train_sag_regressor(ALL_TRAINS[idx_try], PRICES_TRAIN, ALL_VALIDS[idx_try], PRICES_VALID, square_loss,
                                  square_derivative, square_derivative_bias)
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
ACCURACY_TEST = sag_regressor_test(ALL_TRAINS[IDX_TRY_OPT], PRICES_TRAIN, ALL_TESTS[IDX_TRY_OPT], PRICES_TEST,
                                   square_loss, square_derivative, square_derivative_bias, ETA_OPT, LAMBDA_OPT)

print("The accuracy for the test set is :", ACCURACY_TEST)
print("It was made with the preprocessing :", LIST_PREPROCESS[IDX_TRY_OPT])
print("The optimal value of lambda is :", LAMBDA_OPT)
print("The optimal value of eta is :", ETA_OPT)

# Plot the losses
plt.plot(np.arange(0, NB_TRAININGS, 1), ACCURACIES)
plt.xlabel("Different preprocessing")
plt.ylabel("Validation accuracy")
plt.show()
