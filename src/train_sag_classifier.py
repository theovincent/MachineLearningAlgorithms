from pathlib import Path
import numpy as np
from src.classification.sag_classifier import SAGClassifier
from src.losses.hinge_loss import hinge_derivative
from src.utils.read_csv import read_csv
from src.utils.read_txt import read_txt
from src.utils.standardise import standardize_data
from src.utils.visualisation import visualisation


def train_sag_classifier(samples, labels, loss_derivative, show=False):
    sag_classifier = SAGClassifier(loss_derivative)

    sag_classifier.fit(samples, labels, visualisation, nb_epochs=5)

    ortho_vects = sag_classifier.ortho_memory
    biases = np.zeros(len(ortho_vects))

    if show:
        visualisation(ortho_vects, biases, samples)


if __name__ == "__main__":
    # Get the data
    PATH_CSV = Path("../data/small_data.txt")
    (SMALL_DATA, SMALL_LABEL) = read_txt(PATH_CSV)
    STAND_SMALL_DATA = standardize_data(SMALL_DATA)

    # Train the SAG Classifier
    train_sag_classifier(SMALL_DATA, SMALL_LABEL, hinge_derivative, show=True)
