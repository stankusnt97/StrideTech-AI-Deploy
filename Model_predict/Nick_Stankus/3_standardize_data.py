import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

balanced_data = pd.read_csv("balanced_data.csv")

X_CV = balanced_data[['left_fsr_reading_mv', 'right_fsr_reading_mv', 'hip_distance']] #input variables
Y_CV = balanced_data[['label']] #output

scaler = StandardScaler()
X_CV = scaler.fit_transform(X_CV)

scaled_X_CV = pd.DataFrame(data=X_CV, columns=['left_fsr_reading_mv', 'right_fsr_reading_mv', 'hip_distance'])
scaled_X_CV['label'] = Y_CV.values

print(scaled_X_CV.head())
print(scaled_X_CV.shape)
print(Y_CV.head())

scaled_X_CV.to_csv("scaled_X_CV.csv", index=False)
Y_CV.to_csv("Y_CV.csv", index=False)