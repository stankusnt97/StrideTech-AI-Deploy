from Functions._data_preprocessing import load

data_stand = load('/Users/stankusnt/Desktop/StrideTech/ML Data/standing_nick_stankus.txt', 'Nick_Stankus', 'Standing')
data_sit = load('/Users/stankusnt/Desktop/StrideTech/ML Data/sitting_nick_stankus.txt', 'Nick_Stankus', 'Sitting')
data_walk = load('/Users/stankusnt/Desktop/StrideTech/ML Data/walking_nick_stankus.txt', 'Nick_Stankus', 'Walking')

data = data_stand.append(data_sit)
data = data.append(data_walk)
data.head()

data.corr()

data.shape

data.set_index(['time_stamp', 'left_fsr_reading_mv', 'right_fsr_reading_mv', 'hip_distance', 'left_vibration_trigger', 'right_vibration_trigger','hip_vibration_trigger', 'Name', 'Activity'])

data.info()

data.info()

data.dropna(axis=0)
data.isnull().sum()

data = data[(data.left_fsr_reading_mv != 0) & (data.right_fsr_reading_mv != 0) & (data.hip_distance != 0)]

data.shape

data.info()

data.to_csv("initial_data_cleaned.csv", index=False)