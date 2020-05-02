import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def gaussian(mean, x_coord, y_coord):
    return np.exp(- (x_coord - mean[0]) ** 2 - (y_coord - mean[1]) ** 2)


def generate_gaussian(means, nb_points=100):
    x_coord = np.linspace(-2, 2, nb_points)
    y_coord = np.linspace(-2, 2, nb_points)

    (x_coords, y_coords) = np.meshgrid(x_coord, y_coord)

    z_coord_0 = gaussian(means[0], x_coords, y_coords) + 1
    z_coord_1 = gaussian(means[1], x_coords, y_coords)

    return x_coords, y_coords, z_coord_0 - z_coord_1


def get_data(x_coords, y_coords, z_coords):
    return np.concatenate((np.reshape(x_coords, (-1, 1)), np.reshape(y_coords, (-1, 1))), axis=1), z_coords.reshape(-1)


if __name__ == "__main__":
    # Generate data
    MEANS = np.array([[0, 0], [2, 2]])
    NB_POINTS = 20
    (X_COORDS, Y_COORDS, Z_COORDS) = generate_gaussian(MEANS, NB_POINTS)

    # Get data
    (DATA, LABEL) = get_data(X_COORDS, Y_COORDS, Z_COORDS)
    print("Shape of data", DATA.shape)
    print("Numbers of label", LABEL.shape)

    # Plot data
    AX = Axes3D(plt.figure())
    print("x_coords", X_COORDS)
    print(X_COORDS.shape)
    print("y_coords", Y_COORDS)
    print(Y_COORDS.shape)
    print("heights", Z_COORDS)
    print(Z_COORDS.shape)
    AX.plot_surface(X_COORDS, Y_COORDS, Z_COORDS)
    plt.show()
