import pandas as pd
from datetime import date
import os

test_cleaned_data = pd.read_csv('initial_data_cleaned.csv')

name = test_cleaned_data['Name'].mode().iloc[0]

path = '/Users/stankusnt/Desktop/Work/StrideTech AI Test/Model_predict/' + str(name) + '/' +str(name) + ' predictions ' + str(date.today())

isDir = os.path.isdir(path)

if isDir == True:
    with open('/Users/stankusnt/Desktop/Work/StrideTech AI Test/Model_training/initial_data_cleaned.csv', 'a') as f:
        (test_cleaned_data).to_csv(f, header=False, index=False)
else:
    print(isDir)
    print('Data has not been used for testing yet')