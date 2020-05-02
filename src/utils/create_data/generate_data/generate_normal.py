import numpy as np


def generate_normal(means, nb_points=10):
    data_normal = np.zeros((2 * nb_points, 2))
    labels = np.ones(2 * nb_points)
    labels[nb_points:] *= -1

    for idx_mean in range(2):
        sigma = np.random.uniform(0.2, 0.6)
        # Compute the coordinates
        horizontal_coord = np.random.normal(means[idx_mean, 0], sigma, nb_points)
        vertical_coord = np.random.normal(means[idx_mean, 1], sigma, nb_points)
        # Register the coordinates
        data_normal[nb_points * idx_mean: nb_points * (idx_mean + 1), 0] = horizontal_coord
        data_normal[nb_points * idx_mean: nb_points * (idx_mean + 1), 1] = vertical_coord

    # Shuffle the data
    all_data = np.concatenate((data_normal, np.array([labels]).T), axis=1)
    np.random.shuffle(all_data)

    return all_data[:, : -1], all_data[:, -1]


if __name__ == "__main__":
    MEANS = np.array([[0, 0], [2, 2]])
    NB_POINTS = 2
    (DATA_NORMAL, LABELS) = generate_normal(MEANS, NB_POINTS)
    print(DATA_NORMAL, LABELS)
    print("Labels", sum(LABELS))
