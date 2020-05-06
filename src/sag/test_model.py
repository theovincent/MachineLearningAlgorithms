import numpy as np
import matplotlib.pyplot as plt
from src.sag.network import SAG


def sag_test(samples, labels, test_samples, test_labels, loss, get_accuracy, hyperparameters):
    # Get the hyperparameters
    (add_bias, lambada, eta) = hyperparameters

    # Initialise the sag
    sag_model = SAG(loss, add_bias, lambada, eta)

    # Train the sag
    sag_model.fit(samples, labels, nb_epochs=50)

    # Get the losses
    losses_epoch = sag_model.losses

    # Plot the losses
    plt.plot(np.arange(0, len(losses_epoch), 1), losses_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot()

    return get_accuracy(sag_model, test_samples, test_labels)
