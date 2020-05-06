import numpy as np


def value(samples, labels, ortho, bias=0):
    return np.linalg.norm(labels - (samples @ ortho + bias))


def derive_ortho(samples, labels, ortho, bias=0):
    if len(samples.shape) > 1:
        return - 2 * samples.T @ (labels - (samples @ ortho + bias))
    else:
        return - 2 * samples * (labels - (samples @ ortho + bias))


def derive_bias(samples, labels, ortho, bias=0):
    return - 2 * (labels - (samples @ ortho + bias))


if __name__ == "__main__":
    # Data
    SAMPLES = np.array([[9, 1, 2], [9, 2, 3]])
    LABELS = np.array([-1, 1])

    # Parameter
    ORTHO_VECT = np.array([3, 4, 4])
    BIAS = 4

    print(value(SAMPLES, LABELS, ORTHO_VECT, BIAS))
    print(derive_ortho(SAMPLES[0], LABELS[0], ORTHO_VECT, BIAS))
    print(derive_bias(SAMPLES[0], LABELS[0], ORTHO_VECT, BIAS))
