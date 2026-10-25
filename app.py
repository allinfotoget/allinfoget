import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load('model.sav')

st.title("Advertising Sales Prediction App")
st.write("Enter advertising spend in TV, Radio, and Newspaper to predict Sales.")

# User Inputs
tv = st.number_input("TV Advertising Spend ($)", min_value=0.0, value=100.0, step=1.0)
radio = st.number_input("Radio Advertising Spend ($)", min_value=0.0, value=20.0, step=1.0)
newspaper = st.number_input("Newspaper Advertising Spend ($)", min_value=0.0, value=25.0, step=1.0)

if st.button("Predict Sales"):
    # Create features array matched to training schema
    features = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])
    prediction = model.predict(features)
    st.success(f"Predicted Sales: {prediction[0]:.2f} M")
