import numpy as np


def square_loss(samples, labels, train_samples, alpha, kernel, param_kernel):
    return np.linalg.norm(labels - alpha @ kernel(train_samples, samples, param_kernel)) ** 2
