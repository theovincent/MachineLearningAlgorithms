import numpy as np
import matplotlib.pyplot as plt


def show_loss(model):
    # Get the loss
    losses_epoch = model.losses

    # Plot the losses
    plt.plot(np.arange(0, len(losses_epoch), 1), losses_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")


def show_parameters(losses, boxes, param_kernel):
    plt.plot(losses, boxes, label="boxes")
    plt.plot(losses, param_kernel, label="kernel parameter")
    plt.xlabel("Loss")
    plt.legend()
