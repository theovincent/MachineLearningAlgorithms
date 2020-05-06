from pathlib import Path
import numpy as np

# For both models
from src.sag.train_model import sag_train

# For Classification
from src.utils.preprocessing.read_file.read_txt import read_txt
from src.sag.loss import hinge_loss
from src.sag.accuracy.classification_acc import classification_acc
from src.sag.visualisation.classification_visu import classification_visu

# For Regression
from src.utils.create_data.generate_data.generate_gaussian import generate_gaussian, get_data
from src.sag.loss import squared_loss
from src.sag.accuracy.regression_acc import regression_acc
from src.sag.visualisation.regression_visu import regression_visu


YOU_WANT_CLASSIFICATION = True
YOU_WANT_REGRESSION = True

ADD_BIAS = True
VISUALISATION = True


# -- Get small data for classification --
PATH_CSV = Path("../data/normal_data.txt")
(DATA_TRAIN, LABEL_TRAIN, DATA_VALID, LABEL_VALID) = read_txt(PATH_CSV)


# -- Get small data for regression --
MEANS = np.array([[0, 0], [1, 1]])
NB_POINTS = 10
(X_COORDS, Y_COORDS, Z_COORDS) = generate_gaussian(MEANS, NB_POINTS)
(DATA, LABEL) = get_data(X_COORDS, Y_COORDS, Z_COORDS)


# -- Set the options --
OPTIONS = [ADD_BIAS, VISUALISATION]


# -- The SAG Classifier --
if YOU_WANT_CLASSIFICATION:
    FUNCTIONS_CLASS = [hinge_loss, classification_acc, classification_visu]
    PARAM_CLASS = np.array([[0.0006, 0.009], [0.1, 1]])
    # Training
    RESULTS_CLASS = sag_train(DATA_TRAIN, LABEL_TRAIN, DATA_VALID, LABEL_VALID, FUNCTIONS_CLASS, OPTIONS, PARAM_CLASS)
    print("Validation accuracy for classification", RESULTS_CLASS[0])
    print("The optimal lambda", RESULTS_CLASS[1])
    print("The optimal eta", RESULTS_CLASS[2])


# -- The SAG Regressor --
if YOU_WANT_REGRESSION:
    FUNCTIONS_REG = [squared_loss, regression_acc, regression_visu]
    PARAM_REG = np.array([[0.0009, 0.04], [0.06, 0.2]])
    RESULTS_REG = sag_train(DATA, LABEL, DATA, LABEL, FUNCTIONS_REG, OPTIONS, PARAM_REG)
    print("Validation accuracy for regression", RESULTS_REG[0])
    print("The optimal lambda", RESULTS_REG[1])
    print("The optimal eta", RESULTS_REG[2])


