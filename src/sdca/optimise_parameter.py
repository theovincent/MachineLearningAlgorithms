import numpy as np
from src.sdca.network import SDCA


def optimise_parameter(samples, labels, validation, valid_labels, functions, extrem_params):
    # Get the functions
    (loss, get_step, poly_kernel, kernel, get_accuracy) = functions

    # Parameters
    nb_try = 2

    boxes = np.linspace(extrem_params[0, 0], extrem_params[0, 1], nb_try)
    if poly_kernel:
        params = np.linspace(extrem_params[1, 0], extrem_params[1, 1], nb_try)
    else:
        params = np.linspace(extrem_params[1, 0], extrem_params[1, 1], nb_try)

    # To track the optimum parameters
    max_accuracy = 0
    box_opt = 0
    param_opt = 0
    losses = np.ones(nb_try)
    for idx_try in range(nb_try):
        # Initialise the sdcag
        sdca_model = SDCA(get_step, kernel, loss, boxes[idx_try], params[idx_try])

        # Train the sag
        sdca_model.fit(samples, labels, nb_epochs=1)

        accuracy = get_accuracy(sdca_model, validation, valid_labels)

        losses[idx_try] = loss(validation, valid_labels, samples, labels, kernel, params[idx_try])

        # Update optimal parameters
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            box_opt = boxes[idx_try]
            param_opt = params[idx_try]

    return box_opt, param_opt, boxes, params, losses
