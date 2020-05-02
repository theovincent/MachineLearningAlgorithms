import numpy as np


def square_loss(samples, labels, ortho_vect, bias):
    return np.linalg.norm(labels - (samples @ ortho_vect + bias))


def square_derivative(sample, label, ortho_vect, bias):
    if len(sample.shape) > 1:
        return - 2 * sample.T @ (label - (sample @ ortho_vect + bias))
    else:
        return - 2 * sample * (label - (sample @ ortho_vect + bias))


def square_derivative_bias(sample, label, ortho_vect, bias):
    return - 2 * (label - (sample @ ortho_vect + bias))


if __name__ == "__main__":
    # Data
    SAMPLES = np.array([[9, 1, 2], [9, 2, 3]])
    LABELS = np.array([-1, 1])

    # Parameter
    ORTHO_VECT = np.array([3, 4, 4])
    BIAS = 4

    print(square_loss(SAMPLES, LABELS, ORTHO_VECT, BIAS))
    print(square_derivative(SAMPLES[0], LABELS[0], ORTHO_VECT, BIAS))
    print(square_derivative_bias(SAMPLES[0], LABELS[0], ORTHO_VECT, BIAS))
