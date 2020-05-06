import numpy as np
import random as rd
import matplotlib.pyplot as plt


def generate_square(length, nb_points=10, extreme_values=[0, 1]):
    x_coord = np.linspace(extreme_values[0], extreme_values[1], nb_points)
    x_sample = []
    y_coord = np.ones(nb_points)
    y_sample = []

    for idx in range(nb_points):
        if (x_coord[idx] // length) % 2 == 1:
            y_coord[idx] = -1
        if rd.randint(0, 2) == 0:
            x_sample.append([x_coord[idx]])
            y_sample.append(y_coord[idx])

    return np.reshape(x_coord, (nb_points, 1)), y_coord, np.array(x_sample), np.array(y_sample)


if __name__ == "__main__":
    NB_POINTS = 100
    (X_COORD, Y_COORD, X_SAMPLE, Y_SAMPLE) = generate_square(0.25, nb_points=NB_POINTS)
    plt.plot(np.reshape(X_COORD, NB_POINTS), Y_COORD)
    plt.plot(np.reshape(X_SAMPLE, len(X_SAMPLE)), Y_SAMPLE)
    plt.show()

