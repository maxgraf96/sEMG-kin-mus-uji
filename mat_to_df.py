import pickle

import numpy as np
import pandas as pd
from mat4py import loadmat

if __name__ == "__main__":
    data = loadmat("data/KIN_MUS_UJI.mat")

    # Convert to pandas dataframe
    data = pd.DataFrame(data)

    # Save dataframe to csv
    data.to_csv('data/KIN_MUS_UJI.csv', index=False)

