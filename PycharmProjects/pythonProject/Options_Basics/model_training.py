#######################################################################################################################
# Author: Howard Li
# The below model obtain the features created from option_data.py as input
# The aim is to predict the IV return (tomorrow IV / today IV - 1)
# The model is referenced from Daniel & Arthur 2021's “Deep Learning Based Dynamic Implied Volatility Surface"
# The link to the paper is here: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3952842

# The Model is using ConvLSTM 2D
# The referenced model sample is borrowed from Arthur Book
# Link: https://github.com/ArthurBook/MachineSmiling/blob/main/Predicting_vola_dynamics.ipynb

# 1. Data Preprocessing
#       - Train_Test_Split
#       - Scaling Features: (Skew_25D_1, Skew_25D_2, Skew_25D_3, Skew_10D_1, Skew_10D_2, Skew_10D_3,
#                                        Kurtosis_25D_due_1, Kurtosis_25D_due_2, Kurtosis_25D_due_3,
#                                        ATMVOL_due_1, ATMVOL_due_2, ATMVOL_due_3, next_ATMVOL_due_1)
#       - Target Variable: (IV_return_due_1)
# 2. How are Input and Predictions each scaled?
# 3.
# 4. Continuous Results display using Autoregressive (365 Days as one rolling window)
# 5.

# Notes:
#
#
#######################################################################################################################

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Nadam
from keras.layers import InputLayer,ConvLSTM2D, Conv2D
from keras.layers import BatchNormalization
from keras.activations import tanh
from keras.layers import Reshape
from tensorflow.keras import layers


def create_dataset(dataset, look_back):
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i-look_back:i, :])
        y.append(dataset[i, :])
    X, y = np.array(X), np.array(y)
    return X, y











if __name__ == '__main__':
    df = pd.read_csv(r"C:\Users\ps\PycharmProjects\pythonProject\feature_df.csv")
    # Drop the row with nans
    df = df.iloc[:-1, :]
    df['IV_return_due_1'] = np.log(df['IV_return_due_1'])
    data = df.set_index(df['target_date'])
    data.drop(['target_date'], axis=1)
    X = data.drop(['target_date', 'IV_return_due_1'], axis=1)
    y = np.expand_dims(data['IV_return_due_1'].values, 1)

    # Define the number of time steps to look back
    look_back = 30

    # Split the data into training and testing sets
    train_size = int(len(data) * 0.8)
    train_data, test_data = data.iloc[0:train_size, :], data.iloc[train_size:len(data), :]

    # Scale the data to values between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Reshape the data for ConvLSTM2D
    X_train, y_train = create_dataset(train_data, look_back)
    X_test, y_test = create_dataset(test_data, look_back)

    # Reshape the data for input to ConvLSTM2D
    X_train = X_train.reshape((X_train.shape[0], look_back, 1, 1, X_train.shape[2]))
    y_train = y_train.reshape((y_train.shape[0], 1, 1, y_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], look_back, 1, 1, X_test.shape[2]))
    y_test = y_test.reshape((y_test.shape[0], 1, 1, y_test.shape[1]))