import numpy as np
import matplotlib.pyplot as plt
from src.sag.network import SAG
from src.sag.visualisation.show_figures import show_loss, show_parameters
from src.sag.optimise_parameter import optimise_parameter


def sag_train(samples, labels, validation, valid_labels, functions, options, param):
    # Get the functions
    (loss, get_accuracy, visualisation) = functions

    # Get the options
    (add_bias, visu) = options

    # Train the parameters
    optim = optimise_parameter(samples, labels, validation, valid_labels, loss, get_accuracy, add_bias, param)
    (lambada_opt, eta_opt, etas, lambadas, losses) = optim

    # Initialise the sag
    sag_model = SAG(loss, add_bias, lambada_opt, eta_opt)

    # Train the sag with the best parameters
    sag_model.fit(samples, labels, nb_epochs=50)

    # Plot the loss
    show_loss(sag_model)
    plt.figure()

    # Plot the evolution of the lambadas and the etas
    show_parameters(losses, lambadas, etas)

    # Show the evolution
    if visu:
        # Get the evolution of the trainable parameters
        ortho = sag_model.ortho_memory
        if add_bias:
            biases = sag_model.bias_memory
        else:
            biases = np.zeros(len(ortho))
        # Plot the animation
        visualisation(ortho, biases, samples, labels)

    # Show the plots
    plt.show()

    return get_accuracy(sag_model, validation, valid_labels), lambada_opt, eta_opt
