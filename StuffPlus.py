# app.py

import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained XGBoost model
model = xgb.XGBRegressor()
model.load_model('xgboost_model.json')  # Load the model from the saved file

# Function to preprocess input data
def preprocess_input(features):
    # Scaling the input features using StandardScaler (same as during training)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(np.array(features).reshape(1, -1))
    return scaled_features

# Streamlit app UI
st.title('Pitching Stuff+ Prediction')

# Add input fields for the model's features
rel_speed = st.number_input("Enter Release Speed (RelSpeed):")
spin_rate = st.number_input("Enter Spin Rate (SpinRate):")
induced_vert_break = st.number_input("Enter Induced Vertical Break (InducedVertBreak):")
horz_break = st.number_input("Enter Horizontal Break (HorzBreak):")
rel_side = st.number_input("Enter Release Side (RelSide):")
rel_height = st.number_input("Enter Release Height (RelHeight):")
spin_axis = st.number_input("Enter Spin Axis (SpinAxisNormalized):")
exit_speed = st.number_input("Enter Exit Speed (ExitSpeed):")
exit_angle = st.number_input("Enter Exit Angle (Angle):")

# Feature list based on the model
features = [
    rel_speed, spin_rate, induced_vert_break, horz_break, 
    rel_side, rel_height, spin_axis, exit_speed, exit_angle
]

# When the user clicks 'Predict', process the input and make a prediction
if st.button('Predict'):
    # Preprocess the input data (scale features)
    processed_input = preprocess_input(features)

    # Make prediction
    prediction = model.predict(processed_input)

    # Show the result
    st.write(f"The predicted Stuff+ score is: {prediction[0]:.2f}")
