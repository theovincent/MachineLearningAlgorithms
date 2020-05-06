import numpy as np


def classification_acc(sag_model, samples, labels):
    # Get the accuracy for each sample
    accuracies = np.maximum(0, np.sign(labels * sag_model.predict(samples)))

    # Get the average accuracy
    return np.mean(accuracies)
