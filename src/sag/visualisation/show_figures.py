import numpy as np
import matplotlib.pyplot as plt


def show_loss(model):
    # Get the loss
    losses_epoch = model.losses

    # Plot the losses
    plt.plot(np.arange(0, len(losses_epoch), 1), losses_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")


def show_parameters(losses, lambadas, etas):
    plt.plot(losses, lambadas, label="lambadas")
    plt.plot(losses, etas, label="etas")
    plt.xlabel("Loss")
    plt.legend()