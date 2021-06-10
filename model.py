from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.layers.normalization import BatchNormalization 
from keras.layers import Dropout
from keras.layers import *
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def nor(train):
    train = train.drop(["Date"],axis=1)
    train_norm = MinMaxScaler().fit_transform(train)
    train_norm = pd.DataFrame(train_norm, columns=['Open','High','Low','Volume','Dividends','Close'])
    return train_norm

def build(train,p,f):
    x_train, y_train = [], []
    for c in range(train.shape[0]-f-p):
        y_train.append(np.array(train.iloc[c+p:c+p+f]["Close"]))
    mod = train.drop(["Close"],axis=1)
    for c in range(mod.shape[0]-f-p):
        x_train.append(np.array(mod.iloc[c:c+p]))
    return np.array(x_train), np.array(y_train)

def shuf(x,y):
    np.random.seed(100)
    randomList = np.arange(x.shape[0])
    np.random.shuffle(randomList)
    return x[randomList], y[randomList]

def spd(x,y,rate):
    x_train = x[int(x.shape[0]*rate):]
    y_train = y[int(y.shape[0]*rate):]
    x_val = x[:int(x.shape[0]*rate)]
    y_val = y[:int(y.shape[0]*rate)]
    return x_train, y_train, x_val, y_val

def mmstack(shape):
    model = Sequential()
    model.add(LSTM(10, input_length=shape[1], input_dim=shape[2], return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss="mse", optimizer="adam")
    return model
