import numpy as np


def value(samples, labels, ortho, bias=0):
    return np.mean(np.maximum(0, 1 - labels * (samples @ ortho + bias)))


def derive_ortho(samples, labels, ortho, bias=0):
    return - labels * samples


def derive_bias(samples, labels, ortho, bias=0):
    return - labels
