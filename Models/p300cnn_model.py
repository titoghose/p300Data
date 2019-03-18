from keras.layers import Input, Conv1D, Conv2D, BatchNormalization, Dropout, Flatten, Dense, Reshape
from keras.optimizers import Adam
from keras.models import Sequential
import keras.backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint

import numpy as np
import pickle

from imblearn.over_sampling import ADASYN, SMOTE

def recall(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  recall = true_positives / (possible_positives + K.epsilon())
  return recall

def precision(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  return precision
      
def f1(y_true, y_pred):
  
  def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

  def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
  
  precision = precision(y_true, y_pred)
  recall = recall(y_true, y_pred)
  return 2*((precision*recall)/(precision+recall+K.epsilon()))

def createModel(num_channels, num_timesteps):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(num_timesteps, num_channels)))
    model.add(Conv1D(filters=16, kernel_size=(1), padding='valid'))
    model.add(Conv1D(filters=16, kernel_size=20, strides=20, padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten()) 
    model.add(Dense(units=128, activation='tanh'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=128, activation='tanh'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=1, activation='sigmoid'))

    return model

data = pickle.load(open("physionetData.p", "rb"))

X_train, X_test, y_train, y_test = data
del(data)

X_train = np.reshape(X_train, (-1, 160, 64))
X_test = np.reshape(X_test, (-1, 160, 64))

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print("0:", len(np.where(y_train == 0)[0]), "\n1:", len(np.where(y_train == 1)[0]))

model = createModel(64, 160)
model.summary()

adam = Adam(lr=1e-5, epsilon=1e-8)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', precision, recall, f1])

filepath="Weights/weights-improvement-{epoch:02d}-{val_f1:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_f1', verbose=1, save_best_only=True, mode='max')

tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
model.fit(x=X_train, y=y_train, batch_size=128, epochs=1000, validation_data=(X_test, y_test), callbacks=[tbCallBack, checkpoint])