import numpy as np
from src.sag.network import SAG


def optimise_parameter(samples, labels, validation, valid_labels, loss, get_accuracy, add_bias, param):
    # Parameters
    nb_try = 5

    lambadas = np.linspace(param[0, 0], param[0, 1], nb_try)
    etas = np.linspace(param[1, 0], param[1, 1], nb_try)
    
    # To track the optimum parameters
    max_accuracy = 0
    lambada_opt = 0
    eta_opt = 0
    losses = np.ones(nb_try)
    for idx_try in range(nb_try):
        # Initialise the sag
        sag_model = SAG(loss, add_bias, lambadas[idx_try], etas[idx_try])

        # Train the sag
        sag_model.fit(samples, labels, nb_epochs=2)

        accuracy = get_accuracy(sag_model, validation, valid_labels)

        # Get the loss
        if add_bias:
            losses[idx_try] = loss.value(validation, valid_labels, sag_model.ortho, sag_model.bias)
        if not add_bias:
            losses[idx_try] = loss.value(validation, valid_labels, sag_model.ortho, bias=0)

        # Update optimal parameters
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            lambada_opt = lambadas[idx_try]
            eta_opt = etas[idx_try]

    return lambada_opt, eta_opt, etas, lambadas, losses
