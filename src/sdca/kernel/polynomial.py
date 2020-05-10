import numpy as np


def polynomial_kernel(sample1, sample2, degree):
    return (sample1 @ sample2.T) ** degree


if __name__ == "__main__":
    SAMPLE1 = np.array([[-1, 0], [0, 2]])
    SAMPLE2 = np.array([1, 2])

    KERNEL = polynomial_kernel(SAMPLE1, SAMPLE1, 4)
    print(KERNEL)
