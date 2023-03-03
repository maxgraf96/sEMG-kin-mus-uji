import ast
import os
import numpy as np

import pandas as pd
from tqdm import tqdm

from hyperparameters import SAMPLE_WINDOW_LENGTH, SAMPLE_HOP_SIZE

"""
This script takes the raw data created by hu_2022_mat_to_numpy.py and preprocesses it, creating the processed data that we use
in the DataModule.
"""

# Raw data path
from hu_2022_dataset_downloader import dataset_root

# Path to the processed data coming from the hu_2022_mat_to_numpy.py script - contains all gestures for all subjects in one big array
DATASET_RAW_PATH = os.path.join(dataset_root, "processed", "hu_2022_raw.npy")
DATASET_OUT_PATH = os.path.join(dataset_root, "processed", "hu_2022_processed.npy")

def preprocess_data():
    # Load data from the numpy array
    data = np.load(DATASET_RAW_PATH, allow_pickle=True)

    # The rows of the data matrix represent the temporal axis, and the data from three experiment sessions are saved in order of occurrence. 
    # Each session contains 6 repeated trials, namely three slow trials and three fast trials. 
    # The temporal length of each trial is 10 seconds, and the data sampling rate is 100Hz.
    # "data" is now in shape (4320000, 23) where 4320000 = 20 Subjects * 12 Gestures * 3 sessions * 6 repetitions * 10 seconds * 100Hz
    # And 23 = 8 EMG channels + 15 finger joint angles
    # One subject therefore has 216000 rows
    # One gesture therefore has 18000 rows
    # The access structure is thus subject > gesture > repetition > time

    # Take every three rows and combine them into one row
    for subject in tqdm(range(1, 21, 1)):
        print("----------------------------------")
        print("Processing subject " + str(subject) + "...")
        print("----------------------------------")
        # Get the rows for the current subject
        subject_data = data[(subject - 1) * 216000:subject * 216000]
        for gesture in range(1, 13, 1):
            print("--- Processing gesture " + str(gesture) + "...")
            gesture_data = subject_data[(gesture - 1) * 18000:gesture * 18000]
            # Get the sessions for the current gesture
            for session in range(1, 4, 1):
                print("------ Processing session " + str(session) + "...")
                session_data = gesture_data[(session - 1) * 6000:session * 6000]
                # Get the repetitions for the current session
                for repetition in range(1, 7, 1):
                    print("--------- Processing repetition " + str(repetition) + "...")
                    repetition_data = session_data[(repetition - 1) * 1000:repetition * 1000]
                    # Get the time steps for the current repetition
                    for time_step in range(0, 1000, 100):
                        print("------------ Processing time step " + str(time_step) + "...")
                        time_step_data = repetition_data[time_step : time_step + 100]
                        # Get the EMG and kinematic data for the current time step
                        emg_data = time_step_data[:, 0:8]
                        kinematic_data = time_step_data[:, 8:23]

                        a = 2


if __name__ == "__main__":
    preprocess_data()

