import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Reshape
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras.layers import LSTM, LeakyReLU, CuDNNLSTM
from keras.callbacks import CSVLogger, ModelCheckpoint
import h5py
import os
from keras.backend import set_session
from keras import regularizers
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import csv
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from binance import Client, enums  # To connect to the Binance market data endpoint
import time

API_KEY = "AcX3SfwyTebVnFCj9zQWYiierieriierdumMMMMMMYYYYYYYYYYYY"
API_SECRET = "AcX3SfwyTebVnFCj9zQWYiierieriierdumMMMMMMYYYYYYYYYYYY"
client = Client(API_KEY, API_SECRET)

look_back = 60
now = datetime.now()
now = now.strftime('%Y-%m-%d')
days_count = datetime.now() - timedelta(days = look_back)
days_count = days_count.strftime('%Y-%m-%d')

def unix_to_day(time):
    return datetime.utcfromtimestamp(time/1000)

def get_data(SYMBOL,limit=100):
    data  = {"Date":[],"Open":[],"High":[],"Low":[],"Close":[]}
    candles = client.get_historical_klines(SYMBOL, client.KLINE_INTERVAL_5MINUTE, days_count)

    for candle in candles:
        data["Date"].append(float(candle[0]))
        data["Open"].append(float(candle[1]))
        data["High"].append(float(candle[2]))
        data["Low"].append(float(candle[3]))
        data["Close"].append(float(candle[4]))

    data = pd.DataFrame(data)
    data['Date'] = data['Date'].apply(unix_to_day)
    data = data.set_index(["Date"])
    return data
data = get_data('BTCUSDT')['Close']


def plot_fn(series, time, format="-", start=0,
            end=None, label=None):
    plt.figure(figsize=(10, 6))
    if type(series) is tuple:
        plt.plot(time[start:end], series[start:end], format)
    else:
        plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14, labels=label)
    plt.grid(True)
    plt.show()


@dataclass
class G:
    series = np.array(data)
    time = np.array(np.arange(len(series)))
    SPLIT_TIME = round(0.8 * int(len(data)))
    WINDOW_SIZE = 64
    BATCH_SIZE = 256
    SHUFFLE_BUFFER_SIZE = 1000


def train_val_split(time, series, time_step=G.SPLIT_TIME):
    time_train = time[:time_step]
    series_train = series[:time_step]
    time_valid = time[time_step:]
    series_valid = series[time_step:]

    return time_train, series_train, time_valid, series_valid


def windowed_dataset(series, window_size=G.WINDOW_SIZE, batch_size=G.BATCH_SIZE, shuffle_buffer=G.SHUFFLE_BUFFER_SIZE):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    ds = ds.batch(batch_size).prefetch(1)
    return ds


time_train, series_train, time_valid, series_valid = train_val_split(G.time, G.series)
train_set = windowed_dataset(series_train, window_size=G.WINDOW_SIZE, batch_size=G.BATCH_SIZE,
                             shuffle_buffer=G.SHUFFLE_BUFFER_SIZE)


def create_uncompiled_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=3,
                               strides=1,
                               activation="relu",
                               padding='causal',
                               input_shape=[G.WINDOW_SIZE, 1]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 400)
    ])
    return model


uncompiled_model = create_uncompiled_model()
try:
    uncompiled_model.predict(train_set)
except:
    print("Your current architecture is incompatible with the windowed dataset, try adjusting it.")
else:
    print("Your current architecture is compatible with the windowed dataset! :)")


def adjusted_learning_rate(dataset):
    model = create_uncompiled_model()
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch / 20))
    optimizer = tf.keras.optimizers.SGD(momentum=0.9)

    # Compile the model passing in the appropriate loss
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])

    history = model.fit(dataset, epochs=100, callbacks=[lr_schedule])
    return history


lr_history = adjusted_learning_rate(train_set)

plt.semilogx(lr_history.history['lr'], lr_history.history['loss'])

def create_model():
  model = create_uncompiled_model()
  model.compile(loss = tf.keras.losses.Huber(),
                optimizer =tf.keras.optimizers.SGD(lr=7e-4, momentum=0.9),
                metrics = ['mae'])
  return model

model = create_model()
history = model.fit(train_set, epochs=50)

def forecast_model(model, series, window_size):
  ds = tf.data.Dataset.from_tensor_slices(series)
  ds = ds.window(window_size, shift = 1, drop_remainder = True)
  ds = ds.flat_map(lambda w: w.batch(window_size))
  ds = ds.batch(32).prefetch(1)
  forecast = model.predict(ds)
  return forecast

rnn_forecast = forecast_model(model, G.series, G.WINDOW_SIZE)
rnn_forecast = rnn_forecast[G.SPLIT_TIME - G.WINDOW_SIZE:-1]

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

plt.figure(figsize =(10,6))
plot_series(time_valid, series_valid)
plot_series(time_valid, rnn_forecast)