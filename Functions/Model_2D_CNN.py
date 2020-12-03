import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats

X_train = np.load('/Model_training/X_train.npy')
X_test = np.load('/Model_training/X_test.npy')

input_shape = X_train[0].shape

def cnn_model():
    model = Sequential()
    model.add(Conv2D(16, (1, 1), activation='relu', input_shape=input_shape))
    model.add(Dropout(0.1))

    model.add(Conv2D(32, (1, 1), activation='relu'))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.005), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    return model

