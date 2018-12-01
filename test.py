import keras.datasets
import numpy as np
from sklearn.model_selection import train_test_split
(X_train, y), (X_test, y_test_fine) = keras.datasets.cifar100.load_data(label_mode='fine')
(X_train_coarse, y_coarse), (X_test_coarse, y_test_coarse) = keras.datasets.cifar100.load_data(label_mode='coarse')

print(np.concatenate((y, y_coarse), axis=1))
# X_train, y_train, X_val, y_val = train_test_split(X_train, y, train_size=0.9, test_size=0.1, stratify=y)
# print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
# print(np.count_nonzero(y_val == 1))