from pathlib import Path
import pandas as pd
import numpy as np


def read_csv(path_csv):
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

    return houses, prices


if __name__ == "__main__":
    DATA_CSV = Path("../data/data.csv")
    (HOUSES, PRICES) = read_csv(DATA_CSV)
    print("Houses \n", HOUSES)
    print("Prices \n", PRICES)
