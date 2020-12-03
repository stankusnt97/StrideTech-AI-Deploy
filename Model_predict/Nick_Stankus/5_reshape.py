
import numpy as np
from sklearn.model_selection import train_test_split

X_CV = np.load('framed_X_CV.npy')
Y_CV = np.load('framed_Y_CV.npy')

# Tuple uncoupling for reference during reshape
X_CV_dim1, X_CV_dim2, X_CV_dim3 = X_CV.shape

# Make 3D model
X_CV = X_CV.reshape(X_CV_dim1, X_CV_dim2, X_CV_dim3, 1)

print(X_CV.shape)

np.save('X_CV_reshaped', X_CV)
np.save('Y_CV_reshaped', Y_CV)