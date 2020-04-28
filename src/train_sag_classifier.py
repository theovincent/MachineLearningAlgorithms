from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.classification.sag_classifier import SAGClassifier
from src.losses.hinge_loss import hinge_loss, hinge_derivative
from src.utils.read_file.read_csv import read_csv
from src.utils.read_file.read_txt import read_txt
from src.utils.standardise import standardize_data
from src.utils.visualisation import visualisation


def train_sag_classifier(samples, labels, loss, loss_derivative, add_bias=True, show_loss=False, show_graph=False):
    # Initialise the SAG
    sag_classifier = SAGClassifier(loss, loss_derivative, add_bias)

    # Train the SAG
    sag_classifier.fit(samples, labels, show_loss, show_graph, nb_epochs=50)

    if show_loss:
        # Get the losses
        losses = sag_classifier.losses
        # Plot the losses
        plt.plot(np.arange(0, len(losses), 1), losses)

    if show_graph:
        # Get the evolution of the trainable parameters
        ortho_vects = sag_classifier.ortho_memory
        biases = sag_classifier.bias_memory
        # Plot the animation
        visualisation(ortho_vects, biases, samples, labels)


if __name__ == "__main__":
    # Get the data
    PATH_CSV = Path("../data/normal_data.txt")
    (SMALL_DATA, SMALL_LABEL) = read_txt(PATH_CSV)
    STAND_SMALL_DATA = standardize_data(SMALL_DATA)

    # Train the SAG Classifier
    train_sag_classifier(SMALL_DATA, SMALL_LABEL, hinge_loss, hinge_derivative, show_loss=True, show_graph=True)
