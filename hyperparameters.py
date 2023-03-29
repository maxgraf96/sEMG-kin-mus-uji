# Number of sEMG samples to use for each prediction, expressed in samples from the sEMG sensor (fs = 100 Hz)
SAMPLE_WINDOW_LENGTH = 100  # 1 second
SAMPLE_HOP_SIZE = 100
BATCH_SIZE = 256
SEQ_LEN = 201
DATASET_SHIFT_SIZE = SEQ_LEN - 1
INFORMER_PREDICTION_LENGTH = DATASET_SHIFT_SIZE