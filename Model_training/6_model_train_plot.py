import numpy as np
from datetime import date
from Functions.plot_learningCurve import plot_learningCurve
from Functions.Model_2D_CNN import cnn_model

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

model = cnn_model()

epochs = 25


history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1)

model.save('Functions/CNN_Model_trained ' + str(date.today()))

plot_learningCurve(history, epochs)