import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


model = tf.keras.models.load_model('LSTM_prediction_model.h5')


def prediction(model,input):
    input= np.array(input)
    prediction = model.predict(input)
    return 'Survival' if prediction[0][0] >= 0.5 else 'No Survival'

col = [['d1_sysbp_noninvasive_min',
 'd1_resprate_max',
 'd1_bun_max',
 'd1_heartrate_min',
 'd1_bun_min',
 'ventilated_apache',
 'd1_sysbp_min',
 'd1_spo2_min',
 'apache_4a_icu_death_prob',
 'apache_4a_hospital_death_prob']]

def main():
    st.header('Survival Prediction')

    st.write('This is a simple demo of the Streamlit framework')
    st.write('It demonstrates how to load a model, make predictions, and display the results')
    

    st.subheader('Input the Data')
    st.write('Please input the data below')

    i = st.number_input('d1_sysbp_noninvasive_min',)
    j = st.number_input('d1_resprate_max',)
    k = st.number_input('d1_bun_max',)
    l = st.number_input('d1_heartrate_min',)
    m = st.number_input('d1_bun_min',)
    n = st.number_input('ventilated_apache',)
    o = st.number_input('d1_sysbp_min',)
    p = st.number_input('d1_spo2_min',)
    q = st.number_input('apache_4a_icu_death_prob',)
    r = st.number_input('apache_4a_hospital_death_prob',)
   



    input = np.array([[i,j,k,l,m,n,o,p,q,r]])
    print(type(i))
    print(input)
    
    
    if st.button('Prediction'):
        pred = prediction(model,input)        
        st.success('The Prediction is ' + pred)


if __name__ == '__main__':

    main()

