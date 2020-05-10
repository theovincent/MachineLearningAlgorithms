import numpy as np
import random as rd
import matplotlib.pyplot as plt


def generate_sawtooth(nb_wave, extreme_values=[0, 1], bias=1):
    nb_points = nb_wave * 40
    length = extreme_values[1] / nb_wave
    x_coord = np.linspace(extreme_values[0], extreme_values[1], nb_points)
    x_sample = []
    y_coord = np.ones(nb_points)
    y_sample = []

    for idx in range(nb_points):
        y_coord[idx] = x_coord[idx] - np.floor(x_coord[idx] / length) * length + bias
        if rd.randint(0, 2) == 0:
            x_sample.append([x_coord[idx]])
            y_sample.append(y_coord[idx])

    return np.reshape(x_coord, (nb_points, 1)), y_coord, np.array(x_sample), np.array(y_sample)


if __name__ == "__main__":
    (X_COORD, Y_COORD, X_SAMPLE, Y_SAMPLE) = generate_sawtooth(3)
    plt.plot(np.reshape(X_COORD, len(X_COORD)), Y_COORD)
    plt.plot(np.reshape(X_SAMPLE, len(X_SAMPLE)), Y_SAMPLE)
    plt.show()

