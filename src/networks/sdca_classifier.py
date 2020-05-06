import numpy as np
import random as rd


class SDCAClassifier:
    def __init__(self, kernel, loss, prediction, box=1):
        # Kernel
        self.kernel = kernel
        # Box constraint
        self.box = box
        # Loss function
        self.loss = loss
        # Prediction function
        self.prediction = prediction

        # Steps
        self.step = None
        self.steps = None

        # Samples
        self.is_train = False
        self.samples = None

        # To plot the loss
        self.losses = None

    def fit(self, samples, labels, get_step, register_loss, nb_epochs=5):
        nb_samples = samples.shape[0]
        indexes = np.arange(0, nb_samples, 1)

        # Initialise the model
        if not self.is_train:
            self.is_train = True
        self.samples = samples
        self.step = np.zeros(nb_samples)
        self.steps = np.zeros((nb_epochs * nb_samples, nb_samples))

        # Initialise losses
        if register_loss:
            self.losses = np.zeros(nb_epochs)

        # Counts the number of samples visited
        idx_begin = 0

        for epoch in range(nb_epochs):
            # Shuffle the samples
            rd.shuffle(indexes)
            # Count the number of different visited samples
            for index_sample in range(nb_samples):
                idx = indexes[index_sample]

                # Get the step
                step = get_step(samples, labels[idx], idx, self.kernel, self.step, self.box)

                # Update the steps and the orthogonal vector
                self.step[idx] += step
                self.steps[idx_begin, :] = self.step

                # Update the index from the beginning
                idx_begin += 1

            # Update the loss
            if register_loss:
                self.losses[epoch] = self.loss(samples, labels, samples, self.step, self.kernel)

    def predict(self, samples, average=True):
        if not self.is_train:
            return None

        # Get the first memory time
        nb_sample_visited = self.steps.shape[0]
        time = rd.randint(int(nb_sample_visited / 2), nb_sample_visited + 1)
        if average:
            final_step = np.mean(self.steps[time:, :], axis=0)
        else:
            final_step = self.steps[time, :]

        return self.prediction(samples, self.samples, final_step, self.kernel)
