# Number of sEMG samples to use for each prediction, expressed in samples from the sEMG sensor (fs = 100 Hz)
SAMPLE_WINDOW_LENGTH = 100  # 1 second
SAMPLE_HOP_SIZE = 100
# BATCH_SIZE = 32
BATCH_SIZE = 512
SEQ_LEN = 101
INFORMER_PREDICTION_LENGTH = 100
# Must be otherwise 
DATASET_SHIFT_SIZE = 100

LOSS_INDICES = [0, 1, 3, 4, 7, 8, 11, 12]