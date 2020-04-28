import numpy as np
import random as rd


class SAGClassifier:
    def __init__(self, loss, loss_derivation, add_bias, lambada=0.001, eta=0.01):
        # Trade off
        self.lambada = lambada
        # Optimisation step
        self.eta = eta
        # Loss function
        self.loss = loss
        # Loss derivative function
        self.loss_derivation = loss_derivation

        # Empirical risk
        self.emp_risk_ortho = None
        self.emp_risk_bias = None
        # Orthoganal_vector
        self.ortho_vect = None
        # Bias
        self.add_bias = add_bias
        self.bias = None

        # Visualisation
        self.ortho_memory = None
        self.bias_memory = None

        # To plot the loss
        self.losses = None

        # Number of sample seen
        self.sample_diff = 1

    def fit(self, samples, labels, register_loss, register_visu, nb_epochs=5):
        (nb_samples, nb_features) = samples.shape
        indexes = np.arange(0, nb_samples, 1)

        # Initialise ortho_vect, bias and emp_risk_ortho
        if not self.emp_risk_ortho:
            self.emp_risk_ortho = np.zeros(nb_features)
        if not self.emp_risk_bias:
            self.emp_risk_bias = 0
        if not self.ortho_vect:
            self.ortho_vect = np.zeros(nb_features)
        if not self.bias:
            self.bias = 0

        # Initialise visualisation
        if register_visu:
            self.ortho_memory = np.zeros((nb_epochs * nb_samples, nb_features))
            self.bias_memory = np.zeros(nb_epochs * nb_samples)

        # Initialise losses
        if register_loss:
            self.losses = np.zeros(nb_epochs)

        # Former loss derivative
        loss_derivative_ortho = np.zeros(nb_features)
        loss_derivative_bias = 0

        for epoch in range(nb_epochs):
            # Shuffle the samples
            rd.shuffle(indexes)
            # Count the number of different visited samples
            for index_sample in range(nb_samples):
                idx = indexes[index_sample]

                # Withdraw former loss derivatives
                self.emp_risk_ortho -= loss_derivative_ortho
                self.emp_risk_bias -= loss_derivative_bias

                # Update loss derivatives
                if self.loss(samples[idx, :], labels[idx], self.ortho_vect, self.bias) > 0:
                    loss_derivative_ortho = self.loss_derivation(samples[idx, :], labels[idx])
                    loss_derivative_bias = - labels[idx]
                else:
                    loss_derivative_ortho = 0
                    loss_derivative_bias = 0

                # Add knew loss derivatives
                self.emp_risk_ortho += loss_derivative_ortho
                self.emp_risk_bias += loss_derivative_bias

                # Update the orthogonal vector
                self.ortho_vect = (1 - self.eta * self.lambada) * self.ortho_vect - self.eta * self.emp_risk_ortho / self.sample_diff

                # Update the bias
                self.bias += - self.eta * self.emp_risk_bias / self.sample_diff

                # Update visualisation
                if register_visu:
                    self.ortho_memory[index_sample + epoch * nb_samples, :] = self.ortho_vect
                    self.bias_memory[index_sample + epoch * nb_samples] = self.bias

                # Update sample_diff
                if self.sample_diff < nb_samples:
                    self.sample_diff += 1

            # Update the loss
            if register_loss and self.add_bias:
                self.losses[epoch] = self.loss(samples, labels, self.ortho_vect, self.bias)
            if register_loss and not self.add_bias:
                self.losses[epoch] = self.loss(samples, labels, self.ortho_vect, bias=0)

    def predict(self, sample):
        if not self.ortho_vect:
            return None
        elif self.add_bias:
            return sample @ self.ortho_vect + self.bias
        else:
            return sample @ self.ortho_vect
