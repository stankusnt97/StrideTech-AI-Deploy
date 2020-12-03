import pandas as pd
import numpy as np
import scipy.stats as stats

def get_frames(df, frame_size, hop_size, N_FEATURES):

    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        left_fsr_reading_mv = df['left_fsr_reading_mv'].values[i: i + frame_size]
        right_fsr_reading_mv = df['right_fsr_reading_mv'].values[i: i + frame_size]
        hip_distance = df['hip_distance'].values[i: i + frame_size]

        # Retrieve the most frequently used label in this segment
        label = stats.mode(df['label'][i: i + frame_size])[0][
            0]  # array within an array returned, and only 1, but still have to reference the index
        frames.append([left_fsr_reading_mv, right_fsr_reading_mv, hip_distance])
        labels.append(label)

    # Bring segments into new and improved shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)  # reformat into vector
    labels = np.asarray(labels)

    print(frames)
    print(labels)
    return frames, labels
