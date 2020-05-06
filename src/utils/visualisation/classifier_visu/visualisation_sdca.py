import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def kernel_ex(sample1, sample2, degree=1):
    return (sample1 @ sample2.T) ** degree


def get_prediction(samples, alpha, kernel):
    print(kernel(samples, samples))
    return alpha @ kernel(samples, samples)


class Visualisation:
    def __init__(self, ax, steps, kernel, points, values, samples, labels, y_lims=[-5, 5]):
        # Set the axes
        self.horizontal_coords = np.reshape(samples, -1)
        self.nb_samples = len(self.horizontal_coords)
        self.ax = ax

        # Set the lines
        self.points = points
        self.values = values
        self.function = ax.plot(np.reshape(points, -1), values, 'green')[0]

        # set the labels
        self.function.set_label("Function")

        # Set up plot parameters
        self.y_lims = y_lims
        self.ax.set_ylim(y_lims[0], y_lims[1])
        self.ax.grid(True)

        # Set the sample
        self.nb_points = len(samples)
        self.ax.scatter(self.horizontal_coords, np.zeros(self.nb_points), color='blue', label="Samples")
        self.scatter_pred = None
        self.scatter_step = None

        # Set the sdca parameters
        self.steps = steps
        self.kernel = kernel
        self.samples = samples
        self.labels = labels

    def init(self):
        return self.function,

    def __call__(self, iteration):
        # Compute the prediction fonction
        vertical_coords = get_prediction(self.samples, self.steps[iteration, :], self.kernel)

        # Erase previous plots
        self.ax.cla()
        self.ax.set_ylim(self.y_lims[0], self.y_lims[1])
        self.ax.grid(True)
        self.function = self.ax.plot(np.reshape(self.points, -1), self.values, 'green')[0]
        self.ax.scatter(self.horizontal_coords, np.zeros(self.nb_points), color='blue', label="Samples")

        # Show the steps
        self.scatter_step = self.ax.scatter(self.horizontal_coords, self.steps[iteration], color='black', label="Alpha")

        # Show the predictions
        self.scatter_pred = self.ax.scatter(self.horizontal_coords, vertical_coords, color='yellow', label="prediction")

        # set the labels
        self.function.set_label("Function")
        self.ax.legend()

        return self.function,


def visualisation(steps, kernel, points, values, samples, labels):
    (fig, ax) = plt.subplots()
    visu = Visualisation(ax, steps, kernel, points, values, samples, labels)
    animation = FuncAnimation(fig, visu, frames=np.arange(0, steps.shape[0], 1), init_func=visu.init, interval=2)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    POINTS = np.array([[0], [0.1], [0.15], [0.2], [0.200001], [0.3], [0.4], [0.5]])
    VALUES = np.array([1, 1, 1, 1, -1, -1, -1, -1])
    SAMPLES = np.array([[0.1], [0.4]])
    LABELS = np.array([1, -1])
    STEPS = np.array([[1, -1]])

    # Plot the regression_visu
    visualisation(STEPS, kernel_ex, POINTS, VALUES, SAMPLES, LABELS)
