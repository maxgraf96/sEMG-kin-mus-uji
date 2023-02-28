import pandas as pd

from data_preprocessing import preprocess_data
from hyperparameters import SAMPLE_WINDOW_LENGTH, SAMPLE_HOP_SIZE


def preprocessing_function(recording, data):
    # Slide a window over the EMG data and add each window to the data, using a hop size of 1
    # As a label, use the corresponding kinematic data of the last sample in the window
    for j in range(0, len(recording['EMG_data']) - SAMPLE_WINDOW_LENGTH, SAMPLE_HOP_SIZE):
        s = {
            'Subject': recording['Subject'],
            'ADL': recording['ADL'],
            'Kinematic_data': recording['Kinematic_data'][j:j + SAMPLE_WINDOW_LENGTH],
            'EMG_data': recording['EMG_data'][j:j + SAMPLE_WINDOW_LENGTH]
        }
        sample = pd.DataFrame.from_dict(s, orient='index').transpose()
        data = pd.concat([data, sample], ignore_index=True)
    
    return data


if __name__ == '__main__':
    preprocess_data(preprocessing_function, name='KIN_MUS_UJI_preprocessed_transformer.pkl')