import matplotlib.pyplot as plt
from src.sdca.network import SDCA
from src.sdca.visualisation.show_figures import show_loss, show_parameters
from src.sdca.optimise_parameter import optimise_parameter


def sdca_train(samples, labels, validation, valid_labels, functions, visu_pack, params):
    # Get the functions
    (loss, get_step, poly_kernel, kernel, get_accuracy) = functions

    # Get the visualisation tools
    (show_plots, show_visu, visualisation, points, values) = visu_pack

    # Train the parameters
    optim = optimise_parameter(samples, labels, validation, valid_labels, functions, params)
    (box_opt, param_opt, boxes, kernel_params, losses) = optim

    # Initialise the sag
    sdca_model = SDCA(get_step, kernel, loss, box_opt, param_opt)

    # Train the sag with the best parameters
    sdca_model.fit(samples, labels, nb_epochs=1)

    # Show the evolution
    if show_visu:
        visualisation(points, values, samples, labels, sdca_model.steps, kernel, param_opt)

    # Show the plots
    if show_plots:
        # Plot the loss
        show_loss(sdca_model)
        plt.figure()
        # Plot the evolution of the lambadas and the etas
        show_parameters(losses, boxes, kernel_params)
        # Show the plots
        plt.show()

    return get_accuracy(sdca_model, validation, valid_labels), box_opt, param_opt

