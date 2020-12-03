import pandas as pd
import numpy as np

from Functions.prepare_frames import get_frames

scaled_X_CV = pd.read_csv("scaled_X_CV.csv")
Y_CV = pd.read_csv("Y_CV.csv")

Fs = 1
frame_size = Fs * 2  # Frames are 0.4 second in length
hop_size = Fs * 1  # New frame created every 0.2 sec
N_FEATURES = 3


X_CV, Y_CV = get_frames(scaled_X_CV, frame_size, hop_size, N_FEATURES)

print(X_CV.shape, Y_CV.shape)

np.save('framed_X_CV.npy', X_CV)
np.save('framed_Y_CV.npy', Y_CV)