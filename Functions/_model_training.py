from sklearn.model_selection import train_test_split
import numpy as np
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam


def split_train_test(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0, stratify = Y) # 80/20 split on train/test, typical split

    print("X_train shape is: " + str(X_train.shape) + "X_test shape is: " + str(X_test.shape))

    # Tuple uncoupling for reference during reshape
    X_train_dim1, X_train_dim2, X_train_dim3 = X_train.shape
    X_test_dim1, X_test_dim2, X_test_dim3 = X_test.shape

    # Make 3D model
    X_train = X_train.reshape(X_train_dim1, X_train_dim2, X_train_dim3, 1)
    X_test = X_test.reshape(X_test_dim1, X_test_dim2, X_test_dim3, 1)
    return X_train, X_test, y_train, y_test

def cnn_model(X_train, y_train, X_test, y_test, epochs=25):
    input_shape = X_train[0].shape
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

    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1)
    return model, history, epochs


def plot_learningCurve(history, epochs):
    # accuracy
    epoch_range = range(1, epochs + 1)
    plt.plot(epoch_range, history.history['sparse_categorical_accuracy'])
    plt.plot(epoch_range, history.history['val_sparse_categorical_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # loss
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()


