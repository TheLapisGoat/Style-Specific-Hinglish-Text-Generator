import numpy as np
import tensorflow as tf
from keras.layers import Embedding, Dense, LSTM, Bidirectional, SpatialDropout1D
import pandas as pd
import string

def new_model(vocab_size, seq_length):
  model = tf.keras.Sequential()
  model.add(Embedding(input_dim = vocab_size, output_dim = 64, input_length = seq_length))
  model.add(SpatialDropout1D(0.4))
  model.add(Bidirectional(LSTM(128, return_sequences=True)))
  model.add(Bidirectional(LSTM(128)))
  model.add(Dense(vocab_size - 1, activation = 'softmax'))
  
  model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
  
  return model