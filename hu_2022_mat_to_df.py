import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import scipy.io as sio 
from scipy.fftpack import fft, fftfreq
from scipy import signal
from tqdm import tqdm

# Raw data root path
from hu_2022_dataset_downloader import dataset_root
from hu_2022_mat_to_df_helper import outlierDector, filtfiltEnvelope

# Output path
output_root = dataset_root + os.path.sep + "processed"


def readfiles(s, fd=100,source='myo', ges_num=12):
    """
    Reads data of each gesture from .mat files, resamples, and concatenates them into x and y matrices.
    """
    addr = os.path.join(dataset_root, "S" + str(s).zfill(2))
    fs = 2048  # Sampling Frequency of Data
    start = 0
    interval = 180  # Data Length of one gesture (seconds): 18 trials x 10 seconds
    x = np.linspace(start * fs, (start + interval) * fs, interval * fd, endpoint=False, dtype=int) 
    
    for i, ges in tqdm(enumerate(range(1,ges_num+1))):
        rawdata = sio.loadmat(os.path.join(addr, "G" + str(ges).zfill(2) + '.mat'))

        # Get myo data
        # Try this if model performs poorly
        # data_myo, _ = filtfiltEnvelope(rawdata['Data'][:,80:88] * 0.04,s,ges) 
        data_myo = rawdata['Data'][:, 80:88]
        data_glove = rawdata['Data'][:, 65:80]

        # Their code
        # if source == 'hdEMG':
            # data,_ = filtfiltEnvelope(rawdata['Data'][:,0:65],s,ges) 
        # elif source == 'glove':
            # data = rawdata['Data'][:,65:80]
        # elif source == 'myo':
            # data,_ = filtfiltEnvelope(rawdata['Data'][:,80:88] * 0.04,s,ges) 
        # elif source == 'imu':
            # data = rawdata['Data'][:,88:]
        
        # Down-sampling to fd
        data_myo = data_myo[x, :]
        data_glove = data_glove[x, :]

        if i == 0:
            data_x = data_myo
            data_y = data_glove
        else:
            data_x = np.vstack((data_x, data_myo))
            data_y = np.vstack((data_y, data_glove))
    return data_x, data_y


if __name__ == "__main__":
    # If output directory does not exist, create it
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # Define properties for data
    # Downsample to 100 Hz
    f_downsample = 100
    subject = 1
    x, y = readfiles(subject, fd=f_downsample, source='myo', ges_num=12)
    # Concat x and y along the columns
    data = np.concatenate((x, y), axis=1)

    # Convert to pandas dataframe with no index
    data = pd.DataFrame(data, index=None)
    # The rows of the data matrix represent the temporal axis, and the data from three experiment sessions are saved in order of occurrence. 
    # Each session contains 6 repeated trials, namely three slow trials and three fast trials. 
    # The temporal length of each trial is 10 seconds, and the data sampling rate is 100Hz.
    # "data" is now in shape (216000, 23) where 216000 = 12 Gestures * 3 sessions * 6 repetitions * 10 seconds * 100Hz
    # And 23 = 8 EMG channels + 15 finger joint angles
    # One gesture therefore has 18000 rows
    # The access structure is thus gesture > repetition > time

    

    a = 2
    # Save dataframe to csv
    # data.to_csv('data/KIN_MUS_UJI.csv', index=False)