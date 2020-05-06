import numpy as np


def kernel_ex(sample1, sample2, degree=1):
    return (sample1 @ sample2.T) ** degree


def prediction(samples, train_samples, alpha, kernel):
    return alpha @ kernel(train_samples, samples)


if __name__ == "__main__":
    SAMPLE = np.array([4])
    SAMPLES1 = np.array([[1], [2]])
    SAMPLES2 = np.array([[1], [2], [3]])
    STEPS = np.array([[2, -4, 7], [4, -1, 3]])

    PRED = np.sign(prediction(SAMPLES1, SAMPLES2, STEPS[1], kernel_ex))
    print(PRED)
    print(PRED.shape)
