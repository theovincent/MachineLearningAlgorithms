from pathlib import Path
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from src.networks.sdca_classifier import SDCAClassifier
from src.utils.create_data.generate_data.generate_square import generate_square
from src.utils.visualisation.classifier_visu.visualisation_sdca import visualisation
from src.accuracy.sdca.accuracy_classification import get_accuracy
from src.kernel.polynomial import polynomial_kernel
from src.kernel.gaussian import gaussian_kernel
from src.losses.sdca.hinge_loss import hinge_loss
from src.prediction_function.sdca_function import prediction
from src.networks.steps.hinge_step import get_hinge_step


def sdca_classifier_test(samples, labels, test_samples, test_labels, kernel, loss, prediction,  get_step, box, lambada,
                         show_loss=False):
    # Initialise the sag
    sdca_classifier = SDCAClassifier(kernel, loss, prediction, box, lambada)

    # Train the sag
    sdca_classifier.fit(samples, labels, get_step, register_loss=True, nb_epochs=5)

    if show_loss:
        # Get the losses
        losses_epoch = sdca_classifier.losses
        # Plot the losses
        plt.plot(np.arange(0, len(losses_epoch), 1), losses_epoch)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    return get_accuracy(sdca_classifier, samples, labels, test_samples, test_labels)


def hyperparameter_opt(samples, labels, validation, valid_labels, kernel, loss, prediction, get_step):
    # Parameters
    nb_try = 5
    lambadas = np.linspace(0.01, 0.02, nb_try)
    boxes = np.linspace(1, 5, nb_try)

    max_accuracy = 0
    box_opt = 0
    lambada_opt = 0
    losses = np.ones(nb_try)
    for idx_try in range(nb_try):
        # Initialise the sag
        sdca_classifier = SDCAClassifier(kernel, loss, prediction, boxes[idx_try], lambadas[idx_try])
        # Train the sag
        sdca_classifier.fit(samples, labels, get_step, register_loss=True, nb_epochs=5)

        accuracy = get_accuracy(sdca_classifier, samples, labels, validation, valid_labels)

        # Get the loss
        nb_samples_visted = sdca_classifier.steps.shape[0]
        rd_idx = rd.randint(int(nb_samples_visted / 2), nb_samples_visted - 1)
        losses[idx_try] = loss(validation, valid_labels, samples, labels, sdca_classifier.steps[rd_idx, :], kernel)

        # Update optimal hyperparameters
        if accuracy > max_accuracy:
            lambada_opt = lambadas[idx_try]
            box_opt = boxes[idx_try]
            max_accuracy = accuracy

    return lambada_opt, box_opt, lambadas, boxes, losses


def train_sdca_classifier(full_samples, full_labels, samples, labels, validation, valid_labels, kernel, loss,
                          prediction,  get_step, show_loss=False, show_graph=False, show_hyperparameter=False):
    # Optimize the hyperparameter
    """hyperparameters = optimise_parameter(samples, labels, validation, valid_labels, kernel, loss, prediction, get_step)
    (lambada, box, lambadas, boxes, losses_hyper) = hyperparameters"""

    # Initialise the sag
    sdca_classifier = SDCAClassifier(kernel, loss, prediction)  # ,box, lambada)

    # Train the sag
    sdca_classifier.fit(samples, labels, get_step, register_loss=True, nb_epochs=5)

    if show_loss:
        # Get the losses
        losses_epoch = sdca_classifier.losses
        # Plot the losses
        plt.plot(np.arange(0, len(losses_epoch), 1), losses_epoch)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

    """if show_hyperparameter:
        if show_loss:
            plt.figure()
        plt.plot(losses_hyper, boxes, label="boxes")
        plt.plot(losses_hyper, lambadas, label="lambadas")
        plt.xlabel("Loss")
        plt.legend()"""

    if show_graph:
        # Get the evolution of the trainable parameter
        steps = sdca_classifier.steps
        # Plot the animation
        visualisation(steps, kernel, full_samples, full_labels, samples, labels)

    return get_accuracy(sdca_classifier, validation, valid_labels), 0, 0  # , lambada, box


if __name__ == "__main__":
    # Get the data
    (DATA, LABELS, DATA_TRAIN, LABEL_TRAIN) = generate_square(0.25, nb_points=200)
    (DATA_VALID, LABEL_VALID) = (DATA_TRAIN, LABEL_TRAIN)

    # Train the sag Classifier
    RESULTS = train_sdca_classifier(DATA, LABELS, DATA_TRAIN, LABEL_TRAIN, DATA_VALID, LABEL_VALID, gaussian_kernel,
                                    hinge_loss, prediction, get_hinge_step, show_loss=True, show_graph=True,
                                    show_hyperparameter=True)

    (ACCURACY_VALID, LAMBDA, BOX) = RESULTS

    # Test the sag Classifier
    """ACCURACY_TEST = sdca_classifier_test(DATA_TRAIN, LABEL_TRAIN, DATA_VALID, LABEL_VALID, polynomial_kernel,
                                         hinge_loss, prediction, get_hinge_step, LAMBDA, BOX, show_loss=True)"""

    print("Best validation accuracy", ACCURACY_VALID)
    print("Optimal lambda", LAMBDA)
    print("Optimal box", BOX)
    # print("Test accuracy", ACCURACY_TEST)
