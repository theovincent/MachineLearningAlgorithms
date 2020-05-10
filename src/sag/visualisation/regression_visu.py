import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import ImageMagickWriter


def compute_prediction(samples, ortho, bias, grid_dims):
    prediction = samples @ ortho + bias
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


class RegressionVisu:
    def __init__(self, ax, ortho_vects, biases, samples, labels):
        # Set the axes
        self.ax = ax
        self.ax.view_init(25, 10)
        self.ax.grid(True)

        # Set the surfaces
        self.grid = get_mesh_grid(samples)
        self.heights = get_heights(samples, labels)
        self.function = self.ax.plot_surface(self.grid[0], self.grid[1], self.heights, color='green')
        self.prediction = None

        # Set the sag parameters
        self.samples = samples
        self.dim_grid = self.grid[0].shape
        self.ortho = ortho_vects
        self.biases = biases

    def init(self):
        # Get the heights
        heights = compute_prediction(self.samples, self.ortho[0], self.biases[0], self.dim_grid)
        # Plot the prediction
        self.prediction = self.ax.plot_surface(self.grid[0], self.grid[1], heights, color='black')
        return self.function,

    def __call__(self, iteration):
        # Get the new heights
        heights = compute_prediction(self.samples, self.ortho[iteration], self.biases[iteration], self.dim_grid)

        # Clear the axes
        self.ax.cla()
        self.ax.set_title("Function in black. Prediction in green.")
        # Plot the function to predict
        self.function = self.ax.plot_surface(self.grid[0], self.grid[1], self.heights, color='green')
        # Plot the prediction
        self.prediction = self.ax.plot_surface(self.grid[0], self.grid[1], heights, color='black')

        return self.function,


def regression_visu(ortho, biases, samples, labels):
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.set_title("Function in green. Prediction in black.")
    visu = RegressionVisu(ax, ortho, biases, samples, labels)
    anima = FuncAnimation(fig, visu, frames=np.arange(0, len(ortho), 1), init_func=visu.init, interval=2)
    ax.legend()
    # anima.save('sag_regression.gif', writer=ImageMagickWriter(fps=60, metadata=dict(artist='Me'), bitrate=1800), dpi=100)
    plt.show()


if __name__ == "__main__":
    SAMPLE = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
    LABELS = np.array([1, 2, 2, 4])

    ORTHO = np.array([[4, 5], [-7, 5], [2, -5], [-4, 9]])
    BIASES = np.array([10, 0.9, 0.8, 0.1])

    # Plot the regression_visu
    regression_visu(ORTHO, BIASES, SAMPLE, LABELS)
