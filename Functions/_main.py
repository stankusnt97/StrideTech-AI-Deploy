from Functions._data_preprocessing import load, combine_and_index, new_train_data_merge, balance_data, standardize_data, get_frames, reshape
from Functions._model_training import split_train_test, cnn_model, plot_learningCurve
from datetime import date
import pandas as pd
import glob
import os
import numpy as np
from Functions._model_evaluate import evaluate_model

def model_train(Name, standing_filepath=None, sitting_filepath=None, walking_filepath=None):

    #Initial data clean
    data_stand = load(standing_filepath, Name, Activity='Standing')
    print(data_stand.head())
    data_sit = load(sitting_filepath, Name, Activity='Sitting')
    print(data_sit.head())
    data_walk = load(walking_filepath, Name, Activity='Walking')
    print(data_walk.head())

    train_cleaned_data = combine_and_index(data_stand, data_sit, data_walk)

    balanced_data, label = balance_data(train_cleaned_data)

    scaled_X, Y = standardize_data(balanced_data)

    #Prepare frames

    X, Y = get_frames(scaled_X, Fs=1)


def model_test(Name, standing_filepath=None, sitting_filepath=None, walking_filepath=None, add_to_train_data=False):

    #Initial data clean
    data_stand = load(standing_filepath, Name, Activity='Standing')
    print(data_stand.head())
    data_sit = load(sitting_filepath, Name, Activity='Sitting')
    print(data_sit.head())
    data_walk = load(walking_filepath, Name, Activity='Walking')
    print(data_walk.head())

    train_cleaned_data = combine_and_index(data_stand, data_sit, data_walk)

    balanced_data, label = balance_data(train_cleaned_data)

    scaled_X, Y = standardize_data(balanced_data)

    #Prepare frames

    X, Y = get_frames(scaled_X, Fs=1)

    #Reshape

    X = reshape(X)

    #Evaluate model
    list_of_models = glob.glob('/Users/stankusnt/Desktop/Work/StrideTech AI Test/Models/CNN_Model_trained*')
    latest_model = max(list_of_models, key=os.path.getctime)
    CV_error, predictions, predictions_and_labels = evaluate_model(X_CV=X, Y_CV=Y, model_path=latest_model)

    if add_to_train_data == True:

        # Merge test and train data
        new_train_data_merge(train_cleaned_data)
        data_full = pd.read_csv('/Users/stankusnt/Desktop/Work/StrideTech AI Test/Model_training/initial_data_cleaned.csv')
        balanced_data_full, label_full = balance_data(data_full)
        scaled_X_full, Y_full = standardize_data(balanced_data_full)
        X_full, Y_full = get_frames(scaled_X_full, Fs=1)
        # Split training and test data
        X_train, X_test, y_train, y_test = split_train_test(X_full,Y_full)
        # Train model
        model, history, epochs = cnn_model(X_train, y_train, X_test, y_test)
        # Plot learning curve
        plot_learningCurve(history, epochs)
        # Save model
        model.save('Models/CNN_Model_trained ' + str(date.today()))

    elif add_to_train_data == False:
        print('Did not add to training data')

    return CV_error, predictions, predictions_and_labels