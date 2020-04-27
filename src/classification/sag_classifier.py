import numpy as np
import random as rd


class SAGClassifier:
    def __init__(self, loss_derivation, lambada=0.001, eta=0.01):
        # Trade off
        self.lambada = lambada
        # Optimisation step
        self.eta = eta
        # Loss derivation function
        self.loss_derivation = loss_derivation

        # Empirical risk
        self.emp_risk = None
        # Orthoganal_vector
        self.ortho_vect = None
        # Bias
        self.bias = None

        # Visualisation
        self.ortho_memory = None
        self.bias_memory = None

    def fit(self, samples, labels, visualisation, nb_epochs=5):
        (nb_samples, nb_features) = samples.shape
        indexes = np.arange(0, nb_samples, 1)

        # Shuffle the samples
        rd.shuffle(indexes)

        # Initialise ortho_vect, bias and emp_risk
        if not self.emp_risk:
            self.emp_risk = np.zeros(nb_features)
        if not self.ortho_vect:
            self.ortho_vect = np.zeros(nb_features)
        if not self.bias:
            self.bias = np.zeros(nb_features)

        # Initialise visualisation
        if visualisation:
            self.ortho_memory = np.zeros((nb_epochs * nb_samples, nb_features))
            self.bias_memory = np.zeros(nb_epochs * nb_samples)

        # Former loss derivative
        loss_derivative = np.zeros(nb_features)

        for epoch in range(nb_epochs):
            # Count the number of different visited samples
            sample_diff = 1
            for index_sample in range(nb_samples):
                idx = indexes[index_sample]

                # Update emp_risk
                self.emp_risk -= loss_derivative
                loss_derivative = self.loss_derivation(samples[idx, :], labels[idx])
                self.emp_risk += loss_derivative

                # Update the orthogonal vector
                self.ortho_vect = (1 - self.eta * self.lambada) * self.ortho_vect - self.eta * self.emp_risk / sample_diff

                # Update visualisation
                if visualisation:
                    self.ortho_memory[index_sample + epoch * nb_samples, :] = self.ortho_vect
                    # self.bias

                # Update sample_diff
                if sample_diff < nb_samples:
                    sample_diff += 1

    def predict(self, sample):
        if not self.ortho_vect:
            return None
        return sample @ self.ortho_vect + self.bias
