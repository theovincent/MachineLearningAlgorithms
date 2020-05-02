import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3


def compute_prediction(samples, ortho_vect, bias, grid_dims):
    prediction = samples @ ortho_vect + bias
    return np.reshape(prediction, grid_dims)


def get_mesh_grid(points):
    x_coord = np.unique(points[:, 0])
    y_coord = np.unique(points[:, 1])

    return np.meshgrid(x_coord, y_coord)


def get_heights(points, list_heights):
    nb_points_x = len(np.unique(points[:, 0]))
    nb_points_y = len(np.unique(points[:, 1]))
    heights = np.zeros((nb_points_x, nb_points_y))

    for idx_x in range(nb_points_x):
        for idx_y in range(nb_points_y):
            heights[idx_x, idx_y] = list_heights[idx_x + idx_y * nb_points_x]

    return heights


class Visualisation:
    def __init__(self, ax, ortho_vects, biases, samples, labels):
        # Set the axes
        self.ax = ax

        # Set the surfaces
        self.grid = get_mesh_grid(samples)
        self.heights = get_heights(samples, labels)
        self.function = self.ax.plot_surface(self.grid[0], self.grid[1], self.heights, color='green')
        self.prediction = None

        # Set up plot parameters
        self.ax.view_init(25, 10)
        self.ax.grid(True)

        # Set the SAG parameters
        self.samples = samples
        self.dim_grid = self.grid[0].shape
        self.ortho_vects = ortho_vects
        self.biases = biases

    def init(self):
        # Get the heights
        heights = compute_prediction(self.samples, self.ortho_vects[0], self.biases[0], self.dim_grid)
        # Plot the prediction
        self.prediction = self.ax.plot_surface(self.grid[0], self.grid[1], heights, color='black')
        return self.function,

    def __call__(self, iteration):
        # Get the new heights
        heights = compute_prediction(self.samples, self.ortho_vects[iteration], self.biases[iteration], self.dim_grid)

        # Clear the axes
        self.ax.cla()
        self.ax.set_title("Function in black. Prediction in green.")
        # Plot the function to predict
        self.function = self.ax.plot_surface(self.grid[0], self.grid[1], self.heights, color='green')
        # Plot the prediction
        self.prediction = self.ax.plot_surface(self.grid[0], self.grid[1], heights, color='black')

        return self.function,


def visualisation(ortho_vects, biases, samples, labels):
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.set_title("Function in green. Prediction in black.")
    visu = Visualisation(ax, ortho_vects, biases, samples, labels)
    animation = FuncAnimation(fig, visu, frames=np.arange(0, len(ortho_vects), 1), init_func=visu.init, interval=2)
    plt.show()


if __name__ == "__main__":
    SAMPLE = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
    LABELS = np.array([1, 2, 2, 4])

    ORTHO_VECTS = np.array([[4, 5], [-7, 5], [2, -5], [-4, 9]])
    BIASES = np.array([10, 0.9, 0.8, 0.1])

    # Plot the visualisation
    visualisation(ORTHO_VECTS, BIASES, SAMPLE, LABELS)
