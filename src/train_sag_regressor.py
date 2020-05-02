import numpy as np
import matplotlib.pyplot as plt
from src.classification.sag_regressor import SAGRegressor
from src.losses.square_loss import square_loss, square_derivative, square_derivative_bias
from src.utils.visualisation.regressor_visu.visualisation import visualisation
from src.utils.create_data.generate_data.generate_gaussian import get_data, generate_gaussian
from src.accuracy.accuracy_regression import get_accuracy


def sag_regressor_test(samples, labels, test_samples, test_labels, loss, loss_derivative, loss_derivative_bias,
                       lambada, eta, add_bias=True, show_loss=True):
    # Initialise the SAG
    sag_classifier = SAGRegressor(loss, loss_derivative, loss_derivative_bias, add_bias, lambada=lambada, eta=eta)

    # Train the SAG
    sag_classifier.fit(samples, labels, show_loss, False, nb_epochs=2)

    if show_loss:
        # Get the losses
        losses_epoch = sag_classifier.losses
        # Plot the losses
        plt.plot(np.arange(0, len(losses_epoch), 1), losses_epoch)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    return get_accuracy(sag_classifier, test_samples, test_labels)


def hyperparameter_opt(samples, labels, validation, valid_labels, loss, loss_derivative, loss_derivative_bias,
                       add_bias):
    # Parameters
    nb_try = 5
    lambadas = np.linspace(0.0009, 0.002, nb_try)
    etas = np.linspace(0.009, 0.02, nb_try)

    max_accuracy = 0
    eta_opt = 0
    lambada_opt = 0
    losses = np.ones(nb_try)
    for idx_try in range(nb_try):
        # Initialise the SAG
        sag_regressor = SAGRegressor(loss, loss_derivative, loss_derivative_bias, add_bias, lambada=lambadas[idx_try],
                                      eta=etas[idx_try])
        # Train the SAG
        sag_regressor.fit(samples, labels, register_loss=False, register_visu=False, nb_epochs=1)

        accuracy = get_accuracy(sag_regressor, validation, valid_labels)

        # Get the loss
        if add_bias:
            losses[idx_try] = loss(validation, valid_labels, sag_regressor.ortho_vect, sag_regressor.bias)
        if not add_bias:
            losses[idx_try] = loss(validation, valid_labels, sag_regressor.ortho_vect, bias=0)

        # Update optimal hyperparameters
        if accuracy > max_accuracy:
            lambada_opt = lambadas[idx_try]
            eta_opt = etas[idx_try]
            max_accuracy = accuracy

    return lambada_opt, eta_opt, etas, lambadas, losses


def train_sag_regressor(samples, labels, validation, valid_labels, loss, loss_derivative, loss_derivative_bias,
                        add_bias=True, show_loss=False, show_hyperparameter=False, show_graph=False):
    # Optimize the hyperparameter
    hyperparameters = hyperparameter_opt(samples, labels, validation, valid_labels, loss, loss_derivative,
                                         loss_derivative_bias, add_bias)
    (lambada, eta, etas, lambadas, losses_hyper) = hyperparameters

    # Initialise the SAG
    sag_regressor = SAGRegressor(loss, loss_derivative, loss_derivative_bias, add_bias, lambada=lambada, eta=eta)

    # Train the SAG
    sag_regressor.fit(samples, labels, show_loss, show_graph, nb_epochs=5)

    if show_loss:
        # Get the losses
        losses_epoch = sag_regressor.losses
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
        ortho_vects = sag_regressor.ortho_memory
        if add_bias:
            biases = sag_regressor.bias_memory
        else:
            biases = np.zeros(len(ortho_vects))
        # Plot the animation
        visualisation(ortho_vects, biases, samples, labels)

    return get_accuracy(sag_regressor, validation, valid_labels), lambada, eta


if __name__ == "__main__":
    # Get the data
    MEANS = np.array([[0, 0], [2, 2]])
    NB_POINTS = 200
    (X_COORDS, Y_COORDS, Z_COORDS) = generate_gaussian(MEANS, NB_POINTS)
    (DATA, LABEL) = get_data(X_COORDS, Y_COORDS, Z_COORDS)

    # Normalisation
    # DATA_TRAIN = normalisation_data(DATA_TRAIN)
    # DATA_VALID = normalisation_data(DATA_VALID)

    # Standardisation
    # DATA_TRAIN = standardize_data(DATA_TRAIN)
    # DATA_VALID = standardize_data(DATA_VALID)

    # Train the SAG Classifier
    RESULTS = train_sag_regressor(DATA, LABEL, DATA, LABEL, square_loss, square_derivative,
                                  square_derivative_bias, show_loss=True, show_hyperparameter=True, show_graph=False,
                                  add_bias=True)
    (ACCURACY_VALID, LAMBDA, ETA) = RESULTS

    # Test the SAG Classifier
    ACCURACY_TEST = sag_regressor_test(DATA, LABEL, DATA, LABEL, square_loss, square_derivative,
                                       square_derivative_bias, LAMBDA, ETA, show_loss=True, add_bias=True)

    print("Best validation accuracy", ACCURACY_VALID)
    print("Optimal lambda", LAMBDA)
    print("Optimal eta", ETA)
    print("Test accuracy", ACCURACY_TEST)
