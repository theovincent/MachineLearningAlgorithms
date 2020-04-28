import numpy as np


def hinge_loss(samples, labels, ortho_vect, bias):
    return np.mean(np.maximum(0, 1 - labels * (samples @ ortho_vect + bias)))


def hinge_derivative(sample, label):
    return - label * sample


if __name__ == "__main__":
    # Data
    SAMPLES = np.array([[9, 1, 2], [9, 2, 3]])
    LABELS = np.array([-1, 1])

    # Parameter
    ORTHO_VECT = np.array([3, 4, 4])
    BIAS = 4

    print(hinge_loss(SAMPLES, LABELS, ORTHO_VECT, BIAS))
    print(hinge_derivative(SAMPLES[0], LABELS[0]))
