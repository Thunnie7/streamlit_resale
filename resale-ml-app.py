import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the pre-trained Gradient Boosting Regressor model
model = joblib.load('GBR_model.pkl')

# Set up page title and header
st.title("🏠 House Resale Price Prediction")
st.markdown("This app predicts the resale price of a house based on various input features. Fill in the form below and hit **Predict**!")

# Group inputs in columns
col1, col2, col3 = st.columns(3)

with col1:
    floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=1, max_value=500)
    house_age = st.number_input("House Age (Years)", min_value=0, max_value=99)

with col2:
    year = st.number_input("Year of Sale", min_value=2000, max_value=2025)
    town_encoded = st.number_input("Town (Encoded)", min_value=0, max_value=30)

with col3:
    storey_range_low = st.number_input("Storey Range Low", min_value=0, max_value=1)
    storey_range_medium = st.number_input("Storey Range Medium", min_value=0, max_value=1)
    storey_range_high = st.number_input("Storey Range High", min_value=0, max_value=1)

# One-hot encoded flat types
flat_type_2_room = st.radio("Is it a 2 ROOM?", [0, 1])
flat_type_3_room = st.radio("Is it a 3 ROOM?", [0, 1])
flat_type_4_room = st.radio("Is it a 4 ROOM?", [0, 1])
flat_type_5_room = st.radio("Is it a 5 ROOM?", [0, 1])
flat_type_executive = st.radio("Is it an EXECUTIVE?", [0, 1])
flat_type_multi_generation = st.radio("Is it a MULTI-GENERATION?", [0, 1])

# Prediction button
if st.button("Predict"):
    features = [[
        floor_area_sqm,
        house_age,
        year,
        town_encoded,
        flat_type_2_room,
        flat_type_3_room,
        flat_type_4_room,
        flat_type_5_room,
        flat_type_executive,
        flat_type_multi_generation,
        storey_range_low,
        storey_range_medium,
        storey_range_high
    ]]
    
    # Reshape the feature array to match the model's input shape
    try:
        prediction_log = model.predict(features)
        # Exponentiate the prediction (if the model returns log-transformed price)
        prediction_actual = np.exp(prediction_log)
        st.success(f"The predicted resale price is: **${prediction_actual[0]:,.2f}**")
    except Exception as e:
        st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown("🔖 **Disclaimer**: This is a prototype app. Predictions are for demonstration purposes only.")
