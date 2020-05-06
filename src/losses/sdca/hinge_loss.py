import numpy as np


def hinge_loss(samples, labels, train_samples, alpha, kernel):
    return np.mean(np.maximum(0, 1 - (alpha @ kernel(train_samples, samples) * labels)))
