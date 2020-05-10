import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import ImageMagickWriter


def kernel_ex(sample1, sample2, degree):
    return (sample1 @ sample2.T) ** degree


def get_prediction(samples, alpha, kernel, param):
    return alpha @ kernel(samples, samples, param)


class SDCAVisu:
    def __init__(self, ax, points, values, samples, labels, steps, kernel, param_kernel, y_lims=[-2, 2]):
        # -- For the axes --
        self.ax = ax
        self.ax.set_ylim(y_lims[0], y_lims[1])
        self.y_lims = y_lims
        self.ax.grid(True)

        # -- For the function --
        self.points = np.reshape(points, -1)
        self.values = values
        self.function = ax.plot(self.points, values, 'green')[0]
        self.function.set_label("Function")

        # -- For the model --
        # To plot
        self.x_coords = np.reshape(samples, -1)
        self.nb_samples = len(samples)
        self.ax.scatter(self.x_coords, np.zeros(self.nb_samples), color='blue', label="Samples")
        self.scatter_steps = None
        self.scatter_preds = None
        # To compute
        self.samples = samples
        self.labels = labels
        self.steps = steps
        self.kernel = kernel
        self.param_kernel = param_kernel

    def init(self):
        return self.function,

    def __call__(self, iteration):
        # -- Erase previous plots and initialize the axes --
        self.ax.cla()
        self.ax.set_ylim(self.y_lims[0], self.y_lims[1])
        self.ax.grid(True)
        # Plot the function
        self.function = self.ax.plot(self.points, self.values, 'green')[0]
        self.function.set_label("Function")
        # Plot the samples
        self.ax.scatter(self.x_coords, np.zeros(self.nb_samples), color='blue', label="Samples")

        # -- Show the predictions --
        # Computation
        y_coords = get_prediction(self.samples, self.steps[iteration, :], self.kernel, self.param_kernel)
        # Show the steps
        self.scatter_steps = self.ax.scatter(self.x_coords, self.steps[iteration], color='black', label="Alpha")
        # Show the predictions
        self.scatter_preds = self.ax.scatter(self.x_coords, y_coords, color='yellow', label="prediction")

        # -- Set the labels --
        self.ax.legend()

        return self.function,


def sdca_visu(points, values, samples, labels, steps, kernel, param_kernel):
    (fig, ax) = plt.subplots()
    visu = SDCAVisu(ax, points, values, samples, labels, steps, kernel, param_kernel)
    anima = FuncAnimation(fig, visu, frames=np.arange(0, steps.shape[0], 1), init_func=visu.init, interval=100)
    ax.legend()
    # anima.save('sdca_regression_poly.gif', writer=ImageMagickWriter(fps=60, metadata=dict(artist='Me'), bitrate=1800), dpi=100)
    plt.show()


if __name__ == "__main__":
    # For the function
    POINTS = np.array([[0], [0.1], [0.15], [0.2], [0.200001], [0.3], [0.4], [0.5]])
    VALUES = np.array([1, 1, 1, 1, -1, -1, -1, -1])

    # For the model
    SAMPLES = np.array([[0.1], [0.4]])
    LABELS = np.array([1, -1])

    # The model
    STEPS = np.array([[1, -1], [2, 1]])

    # Parameter for the kernel
    PARAM_KERNEL = 1

    # Plot the regression_visu
    sdca_visu(POINTS, VALUES, SAMPLES, LABELS, STEPS, kernel_ex, PARAM_KERNEL)
