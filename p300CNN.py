from keras.layers import Input, Conv1D, BatchNormalization, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.models import Sequential

from sklearn.metrics import f1_score, recall_score, precision_score

import numpy as np
import pickle
import os

from imblearn.over_sampling import ADASYN, SMOTE


def f1_metric(y_true, y_pred):
    return f1_score(y_true, y_pred)


def precision_metric(y_true, y_pred):
    return precision_score(y_true, y_pred)


def recall_metric(y_true, y_pred):
    return recall_score(y_true, y_pred)


def createModel(num_channels, num_timesteps):
    model = Sequential()
    model.add(Input(shape=(None, num_timesteps, num_channels), name='input'))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=16, kernel_size=(1, num_channels), padding='valid'))
    model.add(Conv1D(filters=16, kernel_size=(num_timesteps/8, 1), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten()) 
    model.add(Dense(units=128, activation='tanh'))
    model.add(Dropout(rate=0.8))
    model.add(Dense(units=128, activation='tanh'))
    model.add(Dropout(rate=0.8))
    model.add(Dense(units=1, activation='sigmoid'))

    return model


def createData(f_name, num_subjects=12):

    if os.path.isfile(f_name):
        return loadData("physionetData.p")

    X_train = np.array([])
    X_test = np.array([])
    y_train = np.array([])
    y_test = np.array([])

    with open(f_name, "rb") as f:
        for i in range(num_subjects):
            data = pickle.load(f)

            if len(data["Train"]["y"]) != 0:
                if X_train.shape[0] == 0:
                    X_train = data["Train"]["X"]
                    y_train = data["Train"]["y"]
                else:
                    X_train = np.vstack((X_train, data["Train"]["X"]))
                    y_train = np.hstack((y_train, data["Train"]["y"]))

            if len(data["Test"]["y"]) != 0:
                if X_test.shape[0] == 0:
                    X_test = data["Test"]["X"]
                    y_test = data["Test"]["y"]
                else:
                    X_test = np.vstack((X_train, data["Test"]["X"]))
                    y_test = np.hstack((y_test, data["Test"]["y"]))
    
    X_train = np.swapaxes(X_train, 1, 2)
    X_test = np.swapaxes(X_test, 1, 2)

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    X_train = np.reshape(X_train, (-1, X_train.shape[1] * X_train.shape[2]))
    X_test = np.reshape(X_test, (-1, X_test.shape[1] * X_test.shape[2]))
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_sample(X_train, y_train)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    with open("physionetData.p", "wb") as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)
        return loadData("physionetData.p")


def loadData(f_name):
    with open(f_name, "rb") as f:
        data = pickle.load(f)
    X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]
    X_train = np.reshape(X_train, (-1, 160, 64))
    X_test = np.reshape(X_test, (-1, 160, 64))
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = createData("/home/upamanyu/Documents/NTU_Creton/EEG_Code/physionetCNNData.pickle")