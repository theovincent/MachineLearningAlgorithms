import numpy as np


def gaussian_kernel(sample1, sample2, gamma):
    if len(sample1.shape) > 1 and len(sample2.shape) > 1:
        result = []
        nb_samples1 = sample1.shape[0]
        for idx_sample in range(nb_samples1):
            result.append(np.exp(- gamma * np.linalg.norm(sample1[idx_sample] - sample2, axis=1) ** 2))

    elif len(sample1.shape) > 1 or len(sample2.shape) > 1:
        result = np.exp(- gamma * np.linalg.norm(sample1 - sample2, axis=1) ** 2)

    else:
        result = np.exp(- gamma * np.linalg.norm(sample1 - sample2) ** 2)

    return np.array(result)


if __name__ == "__main__":
    SAMPLE1 = np.array([[9, 1], [1, 3]])
    SAMPLE2 = np.array([[3, 8]])

    RESULT = gaussian_kernel(SAMPLE1, SAMPLE2, 100)
    print(RESULT)
    print(RESULT.shape)
