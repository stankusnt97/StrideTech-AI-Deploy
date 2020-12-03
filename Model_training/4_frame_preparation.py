import pandas as pd
import numpy as np

from Functions.prepare_frames import get_frames

scaled_X = pd.read_csv("scaled_X.csv")
Y = pd.read_csv("Y.csv")

Fs = 1
frame_size = Fs * 2  # Frames are 0.4 second in length
hop_size = Fs * 1  # New frame created every 0.2 sec
N_FEATURES = 3


X, Y = get_frames(scaled_X, frame_size, hop_size, N_FEATURES)

print(X.shape, Y.shape)

np.save('framed_X.npy', X)
np.save('framed_Y.npy', Y)