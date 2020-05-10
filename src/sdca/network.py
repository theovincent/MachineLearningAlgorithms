import numpy as np
import random as rd


class SDCA:
    def __init__(self, get_step, kernel, loss, box=1, kernel_param=1):
        # Get step
        self.get_step = get_step
        # Kernel
        self.kernel = kernel
        self.param = kernel_param
        # Loss function
        self.loss = loss
        # Box constraint
        self.box = box

        # Steps
        self.step = 0
        self.steps = 0
        self.start_train = False

        # Samples
        self.samples = 0

        # To plot the loss
        self.losses = 0

    def fit(self, samples, labels, nb_epochs=5):
        nb_samples = samples.shape[0]
        indexes = np.arange(0, nb_samples, 1)

        # Initialize the model
        if not self.start_train:
            self.start_train = True
        self.samples = samples
        self.step = np.zeros(nb_samples)
        self.steps = np.zeros((nb_epochs * nb_samples, nb_samples))

        # Initialize losses
        self.losses = np.zeros(nb_epochs)

        for epoch in range(nb_epochs):
            rd.shuffle(indexes)
            for idx_sample in range(nb_samples):
                idx = indexes[idx_sample]

                # Get the step
                step_idx = self.get_step(self.samples, labels[idx], idx, self.kernel, self.param, self.step, self.box)

                # Update the steps
                self.step[idx] += step_idx
                self.steps[idx_sample + epoch * nb_samples, :] = self.step

            # Register the loss
            self.losses[epoch] = self.loss(samples, labels, self.samples, self.step, self.kernel, self.param)

    def predict(self, samples, average=True):
        if not self.start_train:
            return None
        # Get a random time
        nb_sample_visited = self.steps.shape[0]
        time = rd.randint(int(nb_sample_visited / 2), nb_sample_visited - 1)
        if average:
            final_step = np.mean(self.steps[time:, :], axis=0)
        else:
            final_step = self.steps[time, :]

        return final_step @ self.kernel(self.samples, samples, self.param)
