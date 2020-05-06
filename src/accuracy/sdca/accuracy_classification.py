import numpy as np


def get_accuracy(sdca_model, samples, labels):
    # Get the accuracy for each sample
    accuracies = np.maximum(0, np.sign(sdca_model.predict(samples) * labels))

    # Get the average accuracy
    return np.mean(accuracies)
