import numpy as np


def generate_data(means, nb_points=10):
    data_normal = np.zeros((2 * nb_points, 2))
    labels = np.ones(2 * nb_points)
    labels[nb_points:] *= -1

    for idx_mean in range(2):
        sigma = np.random.uniform(0.5, 0.9)
        # Compute the coordinates
        horizontal_coord = np.random.normal(means[idx_mean, 0], sigma, nb_points)
        vertical_coord = np.random.normal(means[idx_mean, 1], sigma, nb_points)
        # Register the coordinates
        data_normal[nb_points * idx_mean: nb_points * (idx_mean + 1), 0] = horizontal_coord
        data_normal[nb_points * idx_mean: nb_points * (idx_mean + 1), 1] = vertical_coord

    return data_normal, labels


if __name__ == "__main__":
    MEANS = np.array([[0, 0], [1, 1]])
    NB_POINTS = 20
    (DATA_NORMAL, LABELS) = generate_data(MEANS, NB_POINTS)
    print("Mean of classes 1", np.mean(DATA_NORMAL[:NB_POINTS, :], axis=0))
    print("Mean of classes 2", np.mean(DATA_NORMAL[NB_POINTS:, :], axis=0))
