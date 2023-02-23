import ast
import os

import pandas as pd
from tqdm import tqdm

from hyperparameters import SAMPLE_WINDOW_LENGTH, SAMPLE_HOP_SIZE

data_dir = "data"


def preprocess_data():
    """
    Takes a csv file created in mat_to_df.py and preprocesses it.
    Preprocessing involves sliding a window over the sEMG data and using the corresponding kinematic data as label.
    :return: None
    """
    # Load data from csv
    df = pd.read_csv(data_dir + os.path.sep + 'KIN_MUS_UJI.csv')
    # Drop unused columns
    df = df.drop(columns=['Phase'])
    df = df.drop(columns=['time'])
    data = pd.DataFrame(columns=df.columns)

    # Take every three rows and combine them into one row
    for i in tqdm(range(0, len(df), 3)):
        recording = {'Subject': df.iloc[i]['Subject'], 'ADL': df.iloc[i]['ADL']}

        # Get the three emg data rows as list
        rec_emg = df.iloc[i:i + 3]['EMG_data'].to_list()
        rec_kin = df.iloc[i:i + 3]['Kinematic_data'].to_list()
        # Parse them from the three strings and convert to one list
        try:
            sublists_emg = [ast.literal_eval(emg) for emg in rec_emg]
            sublists_kin = [ast.literal_eval(kin) for kin in rec_kin]
            recording['EMG_data'] = [item for sublist in sublists_emg for item in sublist]
            recording['Kinematic_data'] = [item for sublist in sublists_kin for item in sublist]

            # Check if length of EMG data is > SAMPLE_WINDOW_LENGTH
            if len(recording['EMG_data']) <= SAMPLE_WINDOW_LENGTH:
                continue

            # Slide a window over the EMG data and add each window to the data, using a hop size of 1
            # As a label, use the corresponding kinematic data of the last sample in the window
            for j in range(0, len(recording['EMG_data']) - SAMPLE_WINDOW_LENGTH, SAMPLE_HOP_SIZE):
                s = {
                    'Subject': recording['Subject'],
                    'ADL': recording['ADL'],
                    'Kinematic_data': recording['Kinematic_data'][j + SAMPLE_WINDOW_LENGTH - 1],
                    'EMG_data': recording['EMG_data'][j:j + SAMPLE_WINDOW_LENGTH]
                }
                sample = pd.DataFrame.from_dict(s, orient='index').transpose()
                data = pd.concat([data, sample], ignore_index=True)

        except:
            continue

    # Save the data to a pickle file
    data.to_pickle(data_dir + os.path.sep + 'KIN_MUS_UJI_preprocessed.pkl')


def load_preprocessed_data():
    """
    Loads the preprocessed data from a pickle file.
    :return: A pandas dataframe containing the preprocessed data
    """
    return pd.read_pickle(data_dir + os.path.sep + 'KIN_MUS_UJI_preprocessed.pkl')


if __name__ == "__main__":
    preprocess_data()

