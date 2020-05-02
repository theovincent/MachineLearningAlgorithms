import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def compute_line(horizontal_coords, ortho_vect, bias, level):
    return -(ortho_vect[0] * horizontal_coords + bias - level) / (ortho_vect[1] + 10 ** -16)


class Visualisation:
    def __init__(self, ax, ortho_vects, biases, points, labels, x_lims=[-5, 5], y_lims=[-5, 5]):
        # Set the axes
        self.horizontal_coords = np.array(x_lims)
        self.ax = ax

        # Set the lines
        self.line_minus1 = ax.plot([], [], 'blue')[0]
        self.line = ax.plot([], [], 'black')[0]
        self.line_1 = ax.plot([], [], 'blue')[0]

        # set the labels
        self.line_minus1.set_label("<x, w> + b = -1")
        self.line.set_label("<x, w> + b = 0")
        self.line_1.set_label("<x, w> + b = 1")

        # Set up plot parameters
        self.ax.set_xlim(x_lims[0], x_lims[1])
        self.ax.set_ylim(y_lims[0], y_lims[1])
        self.ax.grid(True)

        # Set the POINTS
        horizontal_coords_plus = []
        vertical_coords_plus = []
        horizontal_coords_minus = []
        vertical_coords_minus = []
        nb_points = len(points)
        for idx_points in range(nb_points):
            if labels[idx_points] == 1:
                horizontal_coords_plus.append(points[idx_points, 0])
                vertical_coords_plus.append(points[idx_points, 1])
            else:
                horizontal_coords_minus.append(points[idx_points, 0])
                vertical_coords_minus.append(points[idx_points, 1])
        self.ax.scatter(horizontal_coords_plus, vertical_coords_plus, color='green')
        self.ax.scatter(horizontal_coords_minus, vertical_coords_minus, color='red')

        # Set the SAG parameters
        self.ortho_vects = ortho_vects
        self.biases = biases

    def init(self):
        self.line.set_data([], [])
        return self.line,

    def __call__(self, iteration):
        # Compute the lines
        vertical_coords_minus1 = compute_line(self.horizontal_coords, self.ortho_vects[iteration], self.biases[iteration], -1)
        vertical_coords = compute_line(self.horizontal_coords, self.ortho_vects[iteration], self.biases[iteration], 0)
        vertical_coords_1 = compute_line(self.horizontal_coords, self.ortho_vects[iteration], self.biases[iteration], 1)
        # Show the lines
        self.line_minus1.set_data(self.horizontal_coords, vertical_coords_minus1)
        self.line.set_data(self.horizontal_coords, vertical_coords)
        self.line_1.set_data(self.horizontal_coords, vertical_coords_1)

        return self.line,


def visualisation(ortho_vects, biases, points, labels):
    (fig, ax) = plt.subplots()
    visu = Visualisation(ax, ortho_vects, biases, points, labels)
    animation = FuncAnimation(fig, visu, frames=np.arange(0, len(ortho_vects), 1), init_func=visu.init, interval=3)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    POINTS = np.array([[-1, 1], [-4, 1], [-1, 5], [3, 4], [2, 2]])
    LABELS = np.array([1, 1, -1, -1, -1])
    ORTHO_VECTS = np.array([[4, 5], [-7, 5], [2, -5], [-4, 9]])
    BIASES = np.array([10, 0.9, 0.8, 0.1])

    # Plot the visualisation
    visualisation(ORTHO_VECTS, BIASES, POINTS, LABELS)
