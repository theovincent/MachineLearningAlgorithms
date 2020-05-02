from pathlib import Path
import pandas as pd
import numpy as np


def read_csv(path_csv, training_ratio=0.75, testing_ratio=0.6):
    # Read the csv
    data = pd.read_csv(path_csv)

    # Convert the dates
    data['date_int'] = pd.to_datetime(data['date']).dt.strftime("%Y%m%d").astype(int)

    # Get the number of houses in each city
    city = pd.get_dummies(pd.Categorical(data['city']), prefix='city')

    houses = np.concatenate([data[['date_int', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                                   'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_built',
                                   'yr_renovated']].to_numpy(), city.to_numpy()], axis=1)

    # Get the prices
    prices = data['price'].to_numpy()

    # Split the data into two groups : idx_try set and validation set
    nb_houses = len(houses)
    houses_train = houses[: int(nb_houses * training_ratio)]
    prices_train = prices[: int(nb_houses * training_ratio)]
    houses_validation = houses[int(nb_houses * training_ratio):]
    prices_validation = prices[int(nb_houses * training_ratio):]

    # Split the data into two groups : idx_try set and validation set
    nb_houses_valid = len(houses_validation)
    houses_test = houses_validation[: int(nb_houses_valid * testing_ratio)]
    prices_test = prices_validation[: int(nb_houses_valid * testing_ratio)]
    houses_validation = houses_validation[int(nb_houses_valid * testing_ratio):]
    prices_validation = prices_validation[int(nb_houses_valid * testing_ratio):]

    return houses_train, prices_train, houses_validation, prices_validation, houses_test, prices_test


if __name__ == "__main__":
    DATA_CSV = Path("../../../data/data.csv")
    (HOUSES_TRAIN, PRICES_TRAIN, HOUSES_VALIDATION, PRICES_VALIDATION, HOUSES_TEST, PRICES_TEST) = read_csv(DATA_CSV)
    print("Houses for idx_try ", len(HOUSES_TRAIN))
    print("Houses for validation ", len(HOUSES_VALIDATION))
    print("Houses for testing ", len(HOUSES_TEST))
