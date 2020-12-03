import pandas as pd
from sklearn.preprocessing import LabelEncoder

def balance_data(data, truncate_count):
    #shrink data so same # of rows
    walking = data[data['Activity'] == 'Walking'].head(truncate_count)
    sitting = data[data['Activity'] == 'Sitting'].head(truncate_count)
    standing = data[data['Activity'] == 'Standing']

    balanced_data = pd.concat([walking, sitting, standing])

    balanced_data = balanced_data.drop(['time_stamp', 'left_vibration_trigger', 'right_vibration_trigger', 'hip_vibration_trigger', 'Name'], axis = 1)

    # Set labels
    label = LabelEncoder()
    balanced_data['label'] = label.fit_transform(balanced_data['Activity'])
    return balanced_data, label