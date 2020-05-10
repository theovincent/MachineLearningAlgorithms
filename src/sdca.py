import numpy as np

# For both models
from src.sdca.train_sdca import sdca_train
from src.sdca.kernel.polynomial import polynomial_kernel
from src.sdca.kernel.gaussian import gaussian_kernel
from src.sdca.visualisation.sdca_visu import sdca_visu

# For Classification
from src.utils.create_data.generate_data.generate_square import generate_square
from src.sdca.loss.hinge_loss import hinge_loss
from src.sdca.steps.hinge_step import hinge_step
from src.sdca.accuracy.classification_acc import classification_acc

# For Regression
from src.utils.create_data.generate_data.generate_sawtooth import generate_sawtooth
from src.sdca.loss.square_loss import square_loss
from src.sdca.steps.square_step import square_step
from src.sdca.accuracy.regression_acc import regression_acc


YOU_WANT_CLASSIFICATION = False
YOU_WANT_REGRESSION = True


VISUALISATION = True
POLY_KERNEL = True


# -- Get small data for classification --
(POINTS_CLASS, VALUES_CLASS, TRAIN_CLASS, LABEL_CLASS) = generate_square(0.25, nb_points=200)


# -- Get small data for regression --
(POINTS_REG, VALUES_REG, TRAIN_REG, LABEL_REG) = generate_sawtooth(3)


# -- Set the options --
if POLY_KERNEL:
    KERNEL = polynomial_kernel
else:
    KERNEL = gaussian_kernel


# -- The SAG Classifier --
if YOU_WANT_CLASSIFICATION:
    FUNCTIONS_CLASS = [hinge_loss, hinge_step, POLY_KERNEL, KERNEL, classification_acc]
    # Set the range of the parameters for the optimisation : box, degree or gamma
    if POLY_KERNEL:
        PARAM_CLASS = np.array([[0.1, 3], [0.01, 3]])
    else:
        PARAM_CLASS = np.array([[1, 7], [5, 20]])

    VISU_CLASS = [True, VISUALISATION, sdca_visu, POINTS_CLASS, VALUES_CLASS]  # SHOW_PLOT = True
    # Training
    RESULTS_CLASS = sdca_train(TRAIN_CLASS, LABEL_CLASS, TRAIN_CLASS, LABEL_CLASS, FUNCTIONS_CLASS, VISU_CLASS, PARAM_CLASS)
    print("Validation accuracy for classification", RESULTS_CLASS[0])
    print("The optimal box is ", RESULTS_CLASS[1])
    if POLY_KERNEL:
        print("The optimal degree for the polynomial kernel is ", RESULTS_CLASS[2])
    else:
        print("The optimal gaussian parameter gamma is", RESULTS_CLASS[2])


# -- The SAG Regressor --
if YOU_WANT_REGRESSION:
    FUNCTIONS_REG = [square_loss, square_step, POLY_KERNEL, KERNEL, regression_acc]
    # Set the range of the parameters for the optimisation : box, degree or gamma
    if POLY_KERNEL:
        PARAM_REG = np.array([[1, 3], [0.1, 4]])
    else:
        PARAM_REG = np.array([[4, 7], [10, 30]])
    VISU_REG = [True, VISUALISATION, sdca_visu, POINTS_REG, VALUES_REG]  # SHOW_PLOT = True
    # Training
    RESULTS_REG = sdca_train(TRAIN_REG, LABEL_REG, TRAIN_REG, LABEL_REG, FUNCTIONS_REG, VISU_REG, PARAM_REG)
    print("Validation accuracy for regression", RESULTS_REG[0])
    print("The optimal box is ", RESULTS_REG[1])
    if POLY_KERNEL:
        print("The optimal degree for the polynomial kernel is ", RESULTS_REG[2])
    else:
        print("The optimal gaussian parameter gamma is ", RESULTS_REG[2])
