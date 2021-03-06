import numpy as np
import pandas as pd
from tensorflow import keras
from datetime import date
from sklearn.preprocessing import LabelEncoder


X_CV = np.load('X_CV_reshaped.npy')
Y_CV = np.load('Y_CV_reshaped.npy')

trained_model = keras.models.load_model('/Users/stankusnt/Desktop/Work/StrideTech AI Test/Model_training/Functions/CNN_Model_trained ' + str(date.today()))

CV_error_rate = trained_model.evaluate(X_CV, Y_CV, verbose=1)

print("The loss for the test data set is: {} and the accuracy for the test data set is: {:.0%}".format(CV_error_rate[0], CV_error_rate[1]))

predictions = trained_model.predict(X_CV)

data = pd.read_csv("initial_data_cleaned.csv")
name = data['Name'].mode().iloc[0]

trained_model.save(str(name) + ' predictions ' + str(date.today()))

#reformat predictions to make readable

# Generate arg maxes for predictions

label = LabelEncoder()

classes = np.argmax(predictions, axis=1)

classes = label.fit_transform(classes)

print(classes)


predictions = label.inverse_transform(classes)

predictions_ax1 = predictions.shape[0]

print(predictions_ax1)

predictions = predictions.reshape(predictions_ax1, 1)
predictions = pd.DataFrame(data=predictions, columns=['Activity_prediction'])
Y_CV = label.inverse_transform(Y_CV)
Y_CV = pd.DataFrame(data=Y_CV, columns=['Activity'])
predictions_and_labels = pd.merge(Y_CV, predictions, left_index=True, right_index=True)

#Write to CSV
predictions_and_labels.to_csv('predictions_and_labels.csv')

#Compare # of matches between labels and predictions
comparison_column = np.where(predictions_and_labels['Activity'] == predictions_and_labels['Activity_prediction'], True, False)

predictions_and_labels['equal'] = comparison_column
print(predictions_and_labels['equal'].value_counts())
True_count = (predictions_and_labels['equal']==True).sum()
False_count = (predictions_and_labels['equal']==False).sum()

manual_calculated_error_rate = (True_count)/(True_count+False_count)
print("The accuracy for the test data set is: {:.0%}".format(manual_calculated_error_rate))