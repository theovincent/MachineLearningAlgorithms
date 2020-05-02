"""
This module has the purpose of storing a matrix.
"""
from pathlib import Path
import numpy as np
from src.utils.create_data.generate_data.generate_normal import generate_normal
from src.utils.create_data.generate_data.generate_gaussian import generate_gaussian
from src.utils.create_data.generate_data.array_to_string.array_to_string import array_to_string


def store_matrix(matrix, destination_path_name):
    """
    Store the matrix in a .txt file.
    Args:
        matrix (array, one dimension): the array to be stored.

        destination_path_name (string): the complete path where the matrix will be stored.
    """
    with open(destination_path_name, 'w') as file:
        for item in matrix:
            string_row = array_to_string(item)
            string_row += "\n"
            file.write(string_row)


if __name__ == "__main__":
    # --- For the classification ---
    # Get the data
    MEANS = np.array([[0, 0], [2, 2]])
    (DATA, LABELS) = generate_normal(MEANS, nb_points=40)
    MODIFIED_LABELS = np.array([LABELS]).T
    FULL_DATA = np.concatenate((DATA, MODIFIED_LABELS), axis=1)

    # Register the data
    DESTINATION_PATH = Path("../../../data/normal_data.txt")
    store_matrix(FULL_DATA, DESTINATION_PATH)
