from pathlib import Path
import numpy as np
import numpy.linalg as alg
from src.utils.preprocessing.read_file.read_txt import read_txt
from src.utils.preprocessing.read_file.read_csv import read_csv
from src.utils.preprocessing.standardise import standardize_data


def select_eigen(eigen_value, cutoff):
    """
    Selects the eigen vectors of the empirical covariance matrix that are representative.

    Args:
        eigen_value (array): the eigen values.

        cutoff (float): the level where we stop saving the eigen values.

    Returns:
        saved_vector (integer): the number of eigen vector that are selected.
    """
    nb_eigen_vector = len(eigen_value)
    total_sum = sum(eigen_value)
    saved_vector = 0
    sum_eigen_value = 0

    while saved_vector < nb_eigen_vector and sum_eigen_value < cutoff * total_sum:
        sum_eigen_value += eigen_value[saved_vector]
        saved_vector += 1

    # If all the vector have been selected
    if saved_vector == nb_eigen_vector:
        return nb_eigen_vector
    else:
        # If saved_vector = 0, we take the first eigen value
        return max(1, saved_vector - 1)


def pca(data, cutoff, whitening=False):
    """
    Compute the basis that reprensents the data.
    CAUTION : it perform a standardization before applying the pca.

    Args:
        data (array): the train_set in which we have to decrease the dimension.

        cutoff (float): percentage of the sum of the eigen value.

        whitening (bool): indicates the type of the returned pca.

    Returns:
        if whitening = False:
            pca (array): the basis composed of the eigen vectors of the usual pca.

        if whitening = True:
            pca (array): the basis of the whitening pca
        """
    stand_data = standardize_data(data)
    # The covariance matrix
    emp_cov = np.cov(stand_data.transpose())

    # The svd decomposition of the data
    (u_mat, eigen_mat, v_mat) = alg.svd(emp_cov)

    # The number of eigen vectors to keep
    nb_eigen = select_eigen(eigen_mat, cutoff)

    if not whitening:
        # The usual PCA
        return stand_data @ u_mat[:, : nb_eigen]

    else:
        # The whitening PCA
        return stand_data @ (np.diag(1. / np.sqrt(np.diag(eigen_mat[: nb_eigen]) + 10 ** -16)) * u_mat[:, : nb_eigen])


if __name__ == "__main__":
    TXT_PATH = Path("../../../data/normal_data.txt")
    CSV_PATH = Path("../../../data/data.csv")
    CUTOFF = 0.5

    # For txt
    # (DATA_TRAIN, LABEL_TRAIN, DATA_VALID, LABEL_VALID) = read_txt(TXT_PATH)
    # PCA_DATA = pca(DATA_TRAIN, CUTOFF)

    # For csv
    (HOUSES_TRAIN, PRICES_TRAIN, HOUSES_VALIDATION, PRICES_VALIDATION, HOUSES_TEST, PRICES_TEST) = read_csv(CSV_PATH)
    PCA_DATA = pca(HOUSES_TRAIN, CUTOFF, whitening=True)

    print("Number of feature kept for {} % of the data :".format(CUTOFF * 100), PCA_DATA.shape)
