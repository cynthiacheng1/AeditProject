import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import h5py


def load_dataset(path_to_train, path_to_test): 
    train_dataset = h5py.File(path_to_train) 
    train_x = np.array(train_dataset['train_set_x'][:]) 
    train_y = np.array(train_dataset['train_set_y'][:])


test_dataset = h5py.File(path_to_test)
test_x = np.array(test_dataset['test_set_x'][:])
test_y = np.array(test_dataset['test_set_y'][:])

# y reshaped
train_y = train_y.reshape((1, train_x.shape[0]))
test_y = test_y.reshape((1, test_y.shape[0]))

return train_x, train_y, test_x, test_y