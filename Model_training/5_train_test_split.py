
import numpy as np
from sklearn.model_selection import train_test_split

X = np.load('framed_X.npy')
Y = np.load('framed_Y.npy')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0, stratify = Y) # 80/20 split on train/test, typical split

print("X_train shape is: " + str(X_train.shape) + "X_test shape is: " + str(X_test.shape))

# Tuple uncoupling for reference during reshape
X_train_dim1, X_train_dim2, X_train_dim3 = X_train.shape
X_test_dim1, X_test_dim2, X_test_dim3 = X_test.shape

# Make 3D model
X_train = X_train.reshape(X_train_dim1, X_train_dim2, X_train_dim3, 1)
X_test = X_test.reshape(X_test_dim1, X_test_dim2, X_test_dim3, 1)

print(X_train.shape, X_test.shape)

np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
