import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import make_column_transformer

balanced_data = pd.read_csv("balanced_data.csv")

X = balanced_data[['left_fsr_reading_mv', 'right_fsr_reading_mv', 'hip_distance']] #input variables
Y = balanced_data[['label']] #output

transformer = make_column_transformer(
    (StandardScaler(), [
        'left_fsr_reading_mv', 'right_fsr_reading_mv', 'hip_distance'
    ])
)

X = transformer.fit_transform(X)

scaled_X = pd.DataFrame(data=X, columns=['left_fsr_reading_mv', 'right_fsr_reading_mv', 'hip_distance'])
scaled_X['label'] = Y.values

print(scaled_X.head())
print(Y.head())

scaled_X.to_csv("scaled_X.csv", index=False)
Y.to_csv("Y.csv", index=False)