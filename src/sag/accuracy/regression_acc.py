import numpy as np


def regression_acc(sag_model, samples, labels):
    # Get the accuracy for each sample
    accuracies = np.abs(labels - sag_model.predict(samples)) / (labels + 10 ** -16)

    # Withdraw the sample with a 0 label
    nb_samples = len(samples)
    for idx_sample in range(nb_samples):
        if labels[idx_sample] == 0:
            accuracies[idx_sample] = 0

    # Get the average accuracy
    return 1 - np.mean(accuracies)
