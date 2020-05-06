import numpy.linalg as alg


def get_hinge_step(samples, label, idx, kernel, former_step, box):
    to_compare = (1 - label * former_step @ kernel(samples[idx], samples)) / box
    min_value = min(1, to_compare / (kernel(samples[idx], samples[idx]) ** 2 + 10 ** -16) + former_step[idx] * label)
    max_value = max(0, min_value)

    return label * max_value - former_step[idx]
