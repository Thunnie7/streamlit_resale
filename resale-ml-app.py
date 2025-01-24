import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the pre-trained Gradient Boosting Regressor model
model = joblib.load('GBR_model.pkl')

# Manually define the column names based on your training setup
trained_columns = [
    'Floor Area (sqm)', 'House Age (Years)', 'Year of Sale', 'Block Number (Numeric)',
    'Flat Model (Encoded)', 'Town (Encoded)', 'Flat Type 2 Room', 'Flat Type 3 Room', 'Flat Type 4 Room', 'Flat Type 5 Room', 'Flat Type Executive', 'Flat Type Multi Generation','Storey Range Low', 'Storey Range Medium', 'Storey Range High',
    
]

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
    block_numeric = st.number_input("Block Number (Numeric)", min_value=1, max_value=999)

with col3:
    flat_model_encoded = st.number_input("Flat Model (Encoded)", min_value=0, max_value=10)
    town_encoded = st.number_input("Town (Encoded)", min_value=0, max_value=30)

# One-hot encoded flat types
flat_type_2_room = st.radio("Is it a 2 ROOM?", [0, 1])
flat_type_3_room = st.radio("Is it a 3 ROOM?", [0, 1])
flat_type_4_room = st.radio("Is it a 4 ROOM?", [0, 1])
flat_type_5_room = st.radio("Is it a 5 ROOM?", [0, 1])
flat_type_executive = st.radio("Is it an EXECUTIVE?", [0, 1])
flat_type_multi_generation = st.radio("Is it a MULTI-GENERATION?", [0, 1])

# One-hot encoded storey range
storey_range_binned_low = st.radio("Is the storey range Low?", [0, 1])
storey_range_binned_medium = st.radio("Is the storey range Medium?", [0, 1])
storey_range_binned_high = st.radio("Is the storey range High?", [0, 1])

# Prediction button
if st.button("Predict"):
    features = [[
        floor_area_sqm,
        house_age,
        year,
        block_numeric,
        flat_model_encoded,
        town_encoded,
        flat_type_2_room,
        flat_type_3_room,
        flat_type_4_room,
        flat_type_5_room,
        flat_type_executive,
        flat_type_multi_generation,
        storey_range_binned_low,
        storey_range_binned_medium,
        storey_range_binned_high
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
