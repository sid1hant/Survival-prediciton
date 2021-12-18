import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

model = tf.keras.models.load_model('LSTM_prediction_model.h5')

def prediction(model,input):
    input= np.array(input)
    prediction = model.predict_classes(input)
    return prediction
