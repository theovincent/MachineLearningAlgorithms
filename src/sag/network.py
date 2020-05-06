import numpy as np
import random as rd


class SAG:
    def __init__(self, loss, add_bias=True, lambada=0.001, eta=0.01):
        self.loss = loss

        # Parameters to optimize
        self.ortho = 0
        self.ortho_memory = 0
        self.add_bias = add_bias
        self.bias = 0
        self.bias_memory = 0
        self.losses = 0

        # Parameters for SAG
        self.lambada = lambada
        self.eta = eta
        self.start_train = False

    def fit(self, samples, labels, nb_epochs=5):
        # Initialisation
        (nb_samples, nb_features) = samples.shape

        self.ortho_memory = np.zeros((nb_samples * nb_epochs, nb_features))
        self.bias_memory = np.zeros(nb_samples * nb_epochs)
        self.losses = np.zeros(nb_epochs)

        if not self.start_train:
            self.start_train = True
            self.ortho = np.zeros(nb_features)
            emp_risk_ortho = np.zeros(nb_features)
            loss_deriv_ortho = np.zeros(nb_features)
            emp_risk_bias = 0
            loss_deriv_bias = 0

        indexes = np.arange(0, nb_samples, 1)
        nb_view = 1

        for epoch in range(nb_epochs):
            for idx_sample in range(nb_samples):
                rd.shuffle(indexes)
                idx = indexes[idx_sample]

                # Withdraw former loss derivatives
                emp_risk_ortho -= loss_deriv_ortho
                emp_risk_bias -= loss_deriv_bias

                # Update loss derivatives
                if self.loss.value(samples[idx, :], labels[idx], self.ortho, self.bias) > 0:
                    loss_deriv_ortho = self.loss.derive_ortho(samples[idx, :], labels[idx], self.ortho, self.bias)
                    loss_deriv_bias = self.loss.derive_bias(samples[idx, :], labels[idx], self.ortho, self.bias)
                else:
                    loss_deriv_ortho = 0
                    loss_deriv_bias = 0

                # Add knew loss derivatives
                emp_risk_ortho += loss_deriv_ortho
                emp_risk_bias += loss_deriv_bias

                # Update the orthogonal vector
                self.ortho = (1 - self.eta * self.lambada) * self.ortho - self.eta * emp_risk_ortho / nb_view

                # Update the bias
                if self.add_bias:
                    self.bias += - self.eta * emp_risk_bias / nb_view

                # Update regression_visu
                self.ortho_memory[idx_sample + epoch * nb_samples, :] = self.ortho
                self.bias_memory[idx_sample + epoch * nb_samples] = self.bias

                # Update sample_diff
                if nb_view < nb_samples:
                    nb_view += 1

            # Update the loss
            if self.add_bias:
                self.losses[epoch] = self.loss.value(samples, labels, self.ortho, self.bias)
            else:
                self.losses[epoch] = self.loss.value(samples, labels, self.ortho)

    def predict(self, samples):
        if not self.start_train:
            return None
        return samples @ self.ortho + self.bias
