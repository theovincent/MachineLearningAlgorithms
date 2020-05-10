import numpy as np
import matplotlib.pyplot as plt
from src.sdca.network import SDCA


def sdca_test(samples, labels, test_samples, test_labels, functions, parameters):
    # Get the functions
    (loss, get_step, poly_kernel, kernel, get_accuracy) = functions

    # Get the parameters
    (box, param_kernel) = parameters

    # Initialise the sag  (get_step, kernel, loss, box_opt, param_opt)
    sdca_model = SDCA(get_step, kernel, loss, box, param_kernel)

    # Train the sag
    sdca_model.fit(samples, labels, nb_epochs=5)

    # Get the losses
    losses_epoch = sdca_model.losses

    # Plot the losses
    plt.plot(np.arange(0, len(losses_epoch), 1), losses_epoch)
    plt.xlabel("Epoch")
    if poly_kernel:
        plt.ylabel("Loss with polynomial kernel")
    else:
        plt.ylabel("Loss with gaussian kernel")
    plt.plot()

    return get_accuracy(sdca_model, test_samples, test_labels)
