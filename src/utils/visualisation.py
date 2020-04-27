import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def compute_line(horizontal_coords, ortho_vect, bias):
    return -(ortho_vect[0] * horizontal_coords + bias) / ortho_vect[1]


class Visualisation:
    def __init__(self, ax, ortho_vects, biases, points, x_lims=[-5, 5], y_lims=[-5, 5]):
        # Set the axis
        self.line = ax.plot([], [], 'g')[0]
        self.horizontal_coords = np.array(x_lims)
        self.ax = ax

        # Set up plot parameters
        self.ax.set_xlim(x_lims[0], x_lims[1])
        self.ax.set_ylim(y_lims[0], y_lims[1])
        self.ax.grid(True)

        # Set the POINTS
        horizontal_coords = points[:, 0]
        vertical_coords = points[:, 1]
        self.ax.scatter(horizontal_coords, vertical_coords, color='blue')

        # Set the SAG parameters
        self.ortho_vects = ortho_vects
        self.biases = biases

    def init(self):
        self.line.set_data([], [])
        return self.line,

    def __call__(self, iteration):
        # Show the line
        vertical_coords = compute_line(self.horizontal_coords, self.ortho_vects[iteration], self.biases[iteration])
        self.line.set_data(self.horizontal_coords, vertical_coords)

        return self.line,


def visualisation(ortho_vects, biases, points):
    (fig, ax) = plt.subplots()
    visu = Visualisation(ax, ortho_vects, biases, points)
    animation = FuncAnimation(fig, visu, frames=np.arange(0, len(ortho_vects), 1), init_func=visu.init, interval=200)
    plt.show()


if __name__ == "__main__":
    POINTS = np.array([[-1, 1], [-4, 1], [-1, 5], [3, 4], [2, 2]])
    ORTHO_VECTS = np.array([[4, 5], [-7, 5], [2, -5], [-4, 9]])
    BIASES = np.array([10, 0.9, 0.8, 0.1])

    # Plot the visualisation
    visualisation(ORTHO_VECTS, BIASES, POINTS)
