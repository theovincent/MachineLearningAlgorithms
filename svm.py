from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# For the data
from src.utils.preprocess import get_houses_data
from src.utils.preprocessing.split_label import split_label

# For sag method
from src.sag.train_sag import sag_train
from src.sag.test_sag import sag_test
import src.sag.loss.hinge_loss as hinge_sag
from src.sag.accuracy.classification_acc import classification_acc as acc_sag
from src.sag.visualisation.classification_visu import classification_visu as visu_sag

# For sdca method
from src.sdca.train_sdca import sdca_train
from src.sdca.test_sdca import sdca_test
from src.sdca.kernel.polynomial import polynomial_kernel
from src.sdca.kernel.gaussian import gaussian_kernel
from src.sdca.loss.hinge_loss import hinge_loss as hinge_sdca
from src.sdca.steps.hinge_step import hinge_step as step_sdca
from src.sdca.accuracy.classification_acc import classification_acc as acc_sdca
from src.sdca.visualisation.sdca_visu import sdca_visu


YOU_WANT_SAG = False
YOU_WANT_SDCA = True

# -- Set the options --
ADD_BIAS = True
POLY_KERNEL = False


# --- Get the data ---
CSV_PATH = Path("data/data.csv")
(ALL_TRAINS, ALL_VALIDS, ALL_TESTS, PRICES_TRAIN, PRICES_VALID, PRICES_TEST, LIST_PREPROCESS) = get_houses_data(CSV_PATH)

# Split the labels
SPLIT_TRAIN = split_label(PRICES_TRAIN, PRICES_TRAIN)
SPLIT_VALID = split_label(PRICES_VALID, PRICES_TRAIN)
SPLIT_TEST = split_label(PRICES_TEST, PRICES_TRAIN)


# --- SAG ---
# Set the functions, the options and the parameters
FUNCTIONS_SAG = [hinge_sag, acc_sag, visu_sag]
OPTIONS = [ADD_BIAS, False, False]  # [ADD_BIAS, VISUALISATION, SHOW_PLOTS]
PARAM_SAG = np.array([[0.0001, 0.0005], [0.1, 0.5]])  # [LAMBDA, ETA]

if YOU_WANT_SAG:
    # -- Training --
    print("Train the sag...")
    NB_TRAININGS = len(ALL_TRAINS)
    ACCURACIES = np.zeros(NB_TRAININGS)
    ACCURACY_MAX = 0
    LAMBDA_OPT = 0
    ETA_OPT = 0
    IDX_TRY_OPT = None
    for idx_try in range(NB_TRAININGS):
        print(LIST_PREPROCESS[idx_try])

        # Training with the parameters
        RESULTS_SAG = sag_train(ALL_TRAINS[idx_try], SPLIT_TRAIN, ALL_VALIDS[idx_try], SPLIT_VALID, FUNCTIONS_SAG,
                                OPTIONS, PARAM_SAG)

        (ACCURACY_VALID, LAMBDA, ETA) = RESULTS_SAG

        # Update the global parameters
        ACCURACIES[idx_try] = ACCURACY_VALID
        print("Validation accuracy", ACCURACY_VALID)
        if ACCURACY_MAX < ACCURACY_VALID:
            ACCURACY_MAX = ACCURACY_VALID
            LAMBDA_OPT = LAMBDA
            ETA_OPT = ETA
            IDX_TRY_OPT = idx_try

    # -- Testing with the best parameters --
    print("Test the sag...")
    PARAMETERS = [ADD_BIAS, LAMBDA_OPT, ETA_OPT]
    ACCURACY_TEST = sag_test(ALL_TRAINS[IDX_TRY_OPT], SPLIT_TRAIN, ALL_TESTS[IDX_TRY_OPT], SPLIT_TEST, hinge_sag,
                             acc_sag, PARAMETERS)

    print("The accuracy for the test set is :", ACCURACY_TEST)
    print("It was made with the preprocessing :", LIST_PREPROCESS[IDX_TRY_OPT])
    print("The optimal value of lambda is :", LAMBDA_OPT)
    print("The optimal value of eta is :", ETA_OPT)

    # Plot the losses
    plt.figure()
    plt.bar(np.arange(0, NB_TRAININGS, 1), ACCURACIES)
    plt.xlabel("Different preprocessing")
    plt.ylabel("Validation accuracy")
    plt.show()


# --- SDCA ---
# Set the kernel parameters and the functions
if POLY_KERNEL:
    KERNEL = polynomial_kernel
else:
    KERNEL = gaussian_kernel
FUNCTIONS_SDCA = [hinge_sdca, step_sdca, POLY_KERNEL, KERNEL, acc_sdca]
# Set the range of the parameters for the optimisation : box, degree or gamma
if POLY_KERNEL:
    PARAM_SDCA = np.array([[0.1, 3], [1, 5]])
else:
    PARAM_SDCA = np.array([[0.5, 1], [0.1, 1]])  # [BOX, GAMMA]

VISU_SDCA = [False, False, sdca_visu, None, None]  # [SHOW_PLOTS, SHOW_VISU, VISUALISATION, POINTS, VALUES]

if YOU_WANT_SDCA:
    # -- Training --
    print("Train the sdca...")
    NB_TRAININGS = len(ALL_TRAINS)
    ACCURACIES = np.zeros(NB_TRAININGS)
    ACCURACY_MAX = 0
    BOX_OPT = 0
    PARAM_OPT = 0
    IDX_TRY_OPT = None
    for idx_try in range(NB_TRAININGS):
        print(LIST_PREPROCESS[idx_try])

        # Training with the parameters
        RESULTS_SDCA = sdca_train(ALL_TRAINS[idx_try], SPLIT_TRAIN, ALL_VALIDS[idx_try], SPLIT_VALID, FUNCTIONS_SDCA,
                                  VISU_SDCA, PARAM_SDCA)

        (ACCURACY_VALID, BOX, KERNEL_PARAM) = RESULTS_SDCA

        # Update the global parameters
        ACCURACIES[idx_try] = ACCURACY_VALID
        print("Validation accuracy", ACCURACY_VALID)
        if ACCURACY_MAX < ACCURACY_VALID:
            ACCURACY_MAX = ACCURACY_VALID
            BOX_OPT = BOX
            PARAM_OPT = KERNEL_PARAM
            IDX_TRY_OPT = idx_try

    # -- Testing with the best parameters --
    print("Test the sag...")
    PARAMETERS = [BOX_OPT, PARAM_OPT]
    ACCURACY_TEST = sdca_test(ALL_TRAINS[IDX_TRY_OPT], SPLIT_TRAIN, ALL_TESTS[IDX_TRY_OPT], SPLIT_TEST,
                              FUNCTIONS_SDCA, PARAMETERS)

    print("The accuracy for the test set is :", ACCURACY_TEST)
    print("It was made with the preprocessing :", LIST_PREPROCESS[IDX_TRY_OPT])
    print("The optimal value of the box is :", BOX_OPT)
    if POLY_KERNEL:
        print("The optimal degree of the polynomial kernel is :", PARAM_OPT)
    else:
        print("The optimal gamma of the gaussian kernel is :", PARAM_OPT)

    # Plot the losses
    plt.figure()
    plt.bar(np.arange(0, NB_TRAININGS, 1), ACCURACIES)
    plt.xlabel("Different preprocessing")
    plt.ylabel("Validation accuracy")
    plt.show()
