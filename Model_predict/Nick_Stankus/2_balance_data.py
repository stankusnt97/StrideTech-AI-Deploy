import pandas as pd

from Functions.balance_data import balance_data

data = pd.read_csv("initial_data_cleaned.csv")

# *create function to choose length of smallest dataframe for truncating*

walking_data_length = len(data[data['Activity'] == 'Walking'])
sitting_data_length = len(data[data['Activity'] == 'Sitting'])
standing_data_length = len(data[data['Activity'] == 'Standing'])

truncate_count = min(walking_data_length, sitting_data_length, standing_data_length)
balanced_data, label = balance_data(data, truncate_count)

print(label.classes_)
print(balanced_data.head())

balanced_data.to_csv("balanced_data.csv", index=False)