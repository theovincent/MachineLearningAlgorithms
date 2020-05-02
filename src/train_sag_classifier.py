from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.classification.sag_classifier import SAGClassifier
from src.losses.hinge_loss import hinge_loss, hinge_derivative
from src.utils.preprocessing.read_file.read_txt import read_txt
from src.utils.visualisation.classifier_visu.visualisation import visualisation
from src.accuracy.accuracy_classification import get_accuracy


def sag_classifier_test(samples, labels, test_samples, test_labels, loss, loss_derivative, lambada, eta, add_bias=True,
                        show_loss=True):
    # Initialise the SAG
    sag_classifier = SAGClassifier(loss, loss_derivative, add_bias, lambada=lambada, eta=eta)

    # Train the SAG
    sag_classifier.fit(samples, labels, show_loss, False, nb_epochs=50)

    if show_loss:
        # Get the losses
        losses_epoch = sag_classifier.losses
        # Plot the losses
        plt.plot(np.arange(0, len(losses_epoch), 1), losses_epoch)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    return get_accuracy(sag_classifier, test_samples, test_labels)


def hyperparameter_opt(samples, labels, validation, valid_labels, loss, loss_derivative, add_bias):
    # Parameters
    nb_try = 5
    lambadas = np.linspace(0.01, 0.02, nb_try)
    etas = np.linspace(0.1, 0.8, nb_try)

    max_accuracy = 0
    eta_opt = 0
    lambada_opt = 0
    losses = np.ones(nb_try)
    for idx_try in range(nb_try):
        # Initialise the SAG
        sag_classifier = SAGClassifier(loss, loss_derivative, add_bias, lambada=lambadas[idx_try], eta=etas[idx_try])
        # Train the SAG
        sag_classifier.fit(samples, labels, register_loss=False, register_visu=False, nb_epochs=5)

        accuracy = get_accuracy(sag_classifier, validation, valid_labels)

        # Get the loss
        if add_bias:
            losses[idx_try] = loss(validation, valid_labels, sag_classifier.ortho_vect, sag_classifier.bias)
        if not add_bias:
            losses[idx_try] = loss(validation, valid_labels, sag_classifier.ortho_vect, bias=0)

        # Update optimal hyperparameters
        if accuracy > max_accuracy:
            lambada_opt = lambadas[idx_try]
            eta_opt = etas[idx_try]
            max_accuracy = accuracy

    return lambada_opt, eta_opt, etas, lambadas, losses


def train_sag_classifier(samples, labels, validation, valid_labels, loss, loss_derivative, add_bias=True,
                         show_loss=False, show_hyperparameter=False, show_graph=False):
    # Optimize the hyperparameter
    hyperparameters = hyperparameter_opt(samples, labels, validation, valid_labels, loss, loss_derivative, add_bias)
    (lambada, eta, etas, lambadas, losses_hyper) = hyperparameters

    # Initialise the SAG
    sag_classifier = SAGClassifier(loss, loss_derivative, add_bias, lambada=lambada, eta=eta)

    # Train the SAG
    sag_classifier.fit(samples, labels, show_loss, show_graph, nb_epochs=50)

    if show_loss:
        # Get the losses
        losses_epoch = sag_classifier.losses
        # Plot the losses
        plt.plot(np.arange(0, len(losses_epoch), 1), losses_epoch)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

    if show_hyperparameter:
        if show_loss:
            plt.figure()
        plt.plot(losses_hyper, etas, label="etas")
        plt.plot(losses_hyper, lambadas, label="lambadas")
        plt.xlabel("Loss")
        plt.legend()

    if show_graph:
        # Get the evolution of the trainable parameters
        ortho_vects = sag_classifier.ortho_memory
        if add_bias:
            biases = sag_classifier.bias_memory
        else:
            biases = np.zeros(len(ortho_vects))
        # Plot the animation
        visualisation(ortho_vects, biases, samples, labels)

    return get_accuracy(sag_classifier, validation, valid_labels), lambada, eta


if __name__ == "__main__":
    # Get the data
    PATH_CSV = Path("../data/normal_data.txt")
    (DATA_TRAIN, LABEL_TRAIN, DATA_VALID, LABEL_VALID) = read_txt(PATH_CSV)

    # Train the SAG Classifier
    RESULTS = train_sag_classifier(DATA_TRAIN, LABEL_TRAIN, DATA_VALID, LABEL_VALID, hinge_loss, hinge_derivative,
                                   show_loss=True, show_hyperparameter=True, show_graph=True, add_bias=True)
    (ACCURACY_VALID, LAMBDA, ETA) = RESULTS

    # Test the SAG Classifier
    ACCURACY_TEST = sag_classifier_test(DATA_TRAIN, LABEL_TRAIN, DATA_VALID, LABEL_VALID, hinge_loss, hinge_derivative,
                                        LAMBDA, ETA, show_loss=True, add_bias=True)

    print("Best validation accuracy", ACCURACY_VALID)
    print("Optimal lambda", LAMBDA)
    print("Optimal eta", ETA)
    print("Test accuracy", ACCURACY_TEST)
