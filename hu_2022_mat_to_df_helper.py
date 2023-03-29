import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import scipy.io as sio 
from scipy.fftpack import fft, fftfreq
from scipy import signal

T = 1.0 / 2048.0
stopBand = [50.5, 52]
b, a = signal.butter(N=3, Wn=[2*stopBand[0] * T, 2*stopBand[1]* T], btype='bandstop', analog=False)

def outlierDector(raw, s, ges, fs, interval = 180, plot = False,channel_num = 65):
    '''50Hz Norch'''
    raw =  signal.filtfilt(b, a, raw, axis=0)
    
    idxMax = [np.zeros((int(interval * fs * 0.02),)),np.zeros((int(interval * fs * 0.02),))]
    idxMin = [np.zeros((int(interval * fs * 0.02),)),np.zeros((int(interval * fs * 0.02),))]
    iter_num = 0
    # while True: # Until Outlier Ratio < 1% and Maxium Peak < 2 mv
    #     if (len(idxMin[1]) + len(idxMax[1])) < interval * fs * 0.01 and np.max(abs(raw)) < 2:
    #         break
    #     elif (len(idxMin[1]) + len(idxMax[1])) == 0:
    #         break

    #     iter_num += 1
    #     if iter_num % 10 ==0:
    #         print("epoch: %s, Outlier Ratio: %.2f%%, Maxium Peak: %.2f mV" % 
    #               (iter_num ,(len(idxMin[1]) + len(idxMax[1])) / (interval * fs * 0.01), np.max(abs(raw))))
    #     # plt.figure(1)
    #     # plt.plot(raw[:,20],alpha=1)  # np.abs(rawHDsEMG[x,i])
    #     # plt.show()

    #     emgMax = np.mean(raw,axis=1) + 3 * np.std(raw,axis=1)
    #     emgMin = np.mean(raw,axis=1) - 3 * np.std(raw,axis=1)

    #     # print(emgMean.shape,emgMax.shape,emgMin.shape)
    #     idxMax = np.where(raw.T > emgMax) # find outlier above the Maximum
    #     idxMin = np.where(raw.T < emgMin) # find outlier below the Minimum

    #     raw.T[idxMax]=np.nan
    #     raw.T[idxMin]=np.nan
    #     emgMean = np.nanmean(raw.T,axis=0)  # calculate the mean value after filtering the outliers
    #     raw.T[idxMax] = emgMean[idxMax[1]]
    #     raw.T[idxMin] = emgMean[idxMin[1]]
    print("Subject No. %s, Gesture No. %s, epoch: %s, Outlier Ratio: %.2f%%, Maxium Peak: %.2f mV" % 
          (s, ges, iter_num ,(len(idxMin[1]) + len(idxMax[1])) / (interval * fs * 0.01), np.max(abs(raw))))
    return raw


def filtfiltEnvelope(raw,s,ges,fs=2048):    
    '''Outlier Detector with 50Hz Norch & 3 delta Principle'''
    raw = outlierDector(raw,s,ges,fs)
    '''Root Mean Square'''
    duration = int(2048 * 0.4)  # window length of RMS
    tp = np.vstack((np.zeros((duration-1,raw.shape[1])),raw)) ** 2 # add zeros on the front and compute square value of raw data
    rms = np.zeros(np.shape(raw))
    for i in range(raw.shape[0]):
        rms[i,:] = np.mean(tp[i:i+duration,:],axis=0)
    
    return rms, np.abs(raw)