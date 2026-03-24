import streamlit as st
import pickle   
import numpy as np
import pandas as pd

with open('iris_model.pkl','rb') as f:
    model = pickle.load(f)

with open('iris_scaler.pkl','rb') as f:
    scaler = pickle.load(f)

st.title('Iris Flower Species Predictor')
st.write('Enter the details of the iris flower to predict its species')
sepal_length = st.number_input('Sepal Length (cm):', min_value=0.0)
sepal_width = st.number_input('Sepal Width (cm):', min_value=0.0)
petal_length = st.number_input('Petal Length (cm):', min_value=0.0)
petal_width = st.number_input('Petal Width (cm):', min_value=0.0)


if st.button('Predict'):
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_features)
    prediction = model.predict(input_scaled)
    species = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f'The predicted species is: {species[prediction[0]]}')