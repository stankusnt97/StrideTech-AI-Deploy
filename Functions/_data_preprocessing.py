import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import scipy.stats as stats


def load(Filepath=None, Name='N/A', Activity='N/A'):
    if Filepath is not None:

        data = pd.read_csv(str(Filepath), sep ="\s+", header=None)
        data.columns = ['time_stamp', 'left_fsr_reading_mv', 'right_fsr_reading_mv', 'hip_distance',
                              'left_vibration_trigger', 'right_vibration_trigger', 'hip_vibration_trigger']
        data['Name'] = str(Name)
        data['Activity'] = str(Activity)
        data['left_fsr_reading_mv'] = pd.to_numeric(data['left_fsr_reading_mv'], downcast="float")
        data['right_fsr_reading_mv'] = pd.to_numeric(data['right_fsr_reading_mv'], downcast="float")
        data['hip_distance'] = pd.to_numeric(data['hip_distance'], downcast="float")
    else:
        data = pd.DataFrame()
        data.columns = ['time_stamp', 'left_fsr_reading_mv', 'right_fsr_reading_mv', 'hip_distance',
                              'left_vibration_trigger', 'right_vibration_trigger', 'hip_vibration_trigger']
        # Trick to make name and activity values NaN
        data['Name'] = pd.to_numeric(data['Name'], errors='coerce')
        data['Activity'] = pd.to_numeric(data['Activity'], errors='coerce')
        data['left_fsr_reading_mv'] = pd.to_numeric(data['left_fsr_reading_mv'], downcast="float")
        data['right_fsr_reading_mv'] = pd.to_numeric(data['right_fsr_reading_mv'], downcast="float")
        data['hip_distance'] = pd.to_numeric(data['hip_distance'], downcast="float")
    return data

def combine_and_index(data_stand, data_sit, data_walk):
    data = data_stand.append(data_sit)
    data = data.append(data_walk)
    data.set_index(
        ['time_stamp', 'left_fsr_reading_mv', 'right_fsr_reading_mv', 'hip_distance', 'left_vibration_trigger', 'right_vibration_trigger', 'hip_vibration_trigger', 'Name', 'Activity'])
    data.dropna(axis=0)
    data = data[(data.left_fsr_reading_mv != 0) & (data.right_fsr_reading_mv != 0) & (data.hip_distance != 0)]
    print("Initial data clean result: ", data.head())
    return data

def new_train_data_merge(train_cleaned_data):
    with open('/Users/stankusnt/Desktop/Work/StrideTech AI Test/Model_training/initial_data_cleaned.csv', 'a') as f:
        (train_cleaned_data).to_csv(f, header=False, index=False)

def balance_data(data):
    walking_data_length = len(data[data['Activity'] == 'Walking'])
    sitting_data_length = len(data[data['Activity'] == 'Sitting'])
    standing_data_length = len(data[data['Activity'] == 'Standing'])

    truncate_count = min(walking_data_length, sitting_data_length, standing_data_length)

    #shrink data so same # of rows

    walking = data[data['Activity'] == 'Walking'].head(truncate_count)
    sitting = data[data['Activity'] == 'Sitting'].head(truncate_count)
    standing = data[data['Activity'] == 'Standing']

    balanced_data = pd.concat([walking, sitting, standing])

    balanced_data = balanced_data.drop(['time_stamp', 'left_vibration_trigger', 'right_vibration_trigger', 'hip_vibration_trigger', 'Name'], axis = 1)

    # Set labels
    label = LabelEncoder()
    balanced_data['label'] = label.fit_transform(balanced_data['Activity'])
    print("Balanced data result: ", balanced_data.describe())
    return balanced_data, label

def standardize_data(balanced_data):
    X = balanced_data[['left_fsr_reading_mv', 'right_fsr_reading_mv', 'hip_distance']]  # input variables
    Y = balanced_data[['label']]  # output
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    scaled_X = pd.DataFrame(data=X, columns=['left_fsr_reading_mv', 'right_fsr_reading_mv', 'hip_distance'])
    scaled_X['label'] = Y.values
    print("Scaled X description: " + str(scaled_X.describe()))
    return scaled_X, Y

def get_frames(data, Fs=1, frame_size=None, hop_size=None, N_FEATURES=3):
    if frame_size is None:
        frame_size=Fs*2
    if hop_size is None:
        hop_size=Fs*1
    frames = []
    labels = []
    for i in range(0, len(data) - frame_size, hop_size):
        left_fsr_reading_mv = data['left_fsr_reading_mv'].values[i: i + frame_size]
        right_fsr_reading_mv = data['right_fsr_reading_mv'].values[i: i + frame_size]
        hip_distance = data['hip_distance'].values[i: i + frame_size]

        # Retrieve the most frequently used label in this segment
        label = stats.mode(data['label'][i: i + frame_size])[0][
            0]  # array within an array returned, and only 1, but still have to reference the index
        frames.append([left_fsr_reading_mv, right_fsr_reading_mv, hip_distance])
        labels.append(label)

    # Bring segments into new and improved shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)  # reformat into vector
    labels = np.asarray(labels)
    print("Frames sample: " + str(frames[:2,:2]))
    print("Labels sample: " + str(labels[:2]))
    return frames, labels

def reshape(X):
    X_dim1, X_dim2, X_dim3 = X.shape

    # Make 3D Model
    X = X.reshape(X_dim1, X_dim2, X_dim3, 1)
    print("Input shape is: ", X[5].shape)
    return X

