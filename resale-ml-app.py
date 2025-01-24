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
st.set_page_config(page_title="🏠 House Resale Price Prediction", layout="wide")
st.title("🏠 House Resale Price Prediction")
st.markdown("This app predicts the resale price of a house based on various input features. Fill in the form below and hit **Predict**!")

# Sidebar for user inputs
st.sidebar.header('User Input Parameters')

# Create columns for a more organized input form layout
col1, col2, col3 = st.columns(3)

# Column 1 inputs
with col1:
    floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=1, max_value=500, value=50)
    house_age = st.number_input("House Age (Years)", min_value=0, max_value=99, value=30)

# Column 2 inputs
with col2:
    year = st.number_input("Year of Sale", min_value=2000, max_value=2025, value=2020)
    block_numeric = st.number_input("Block Number (Numeric)", min_value=1, max_value=999, value=100)

# Column 3 inputs
with col3:
    flat_model_encoded = st.number_input("Flat Model (Encoded)", min_value=0, max_value=10, value=2)
    town_encoded = st.number_input("Town (Encoded)", min_value=0, max_value=30, value=15)

# One-hot encoded flat types
flat_type_2_room = st.radio("Is it a 2 ROOM?", [0, 1], index=0)
flat_type_3_room = st.radio("Is it a 3 ROOM?", [0, 1], index=0)
flat_type_4_room = st.radio("Is it a 4 ROOM?", [0, 1], index=0)
flat_type_5_room = st.radio("Is it a 5 ROOM?", [0, 1], index=0)
flat_type_executive = st.radio("Is it an EXECUTIVE?", [0, 1], index=0)
flat_type_multi_generation = st.radio("Is it a MULTI-GENERATION?", [0, 1], index=0)

# One-hot encoded storey range
storey_range_binned_low = st.radio("Is the storey range Low?", [0, 1], index=0)
storey_range_binned_medium = st.radio("Is the storey range Medium?", [0, 1], index=1)
storey_range_binned_high = st.radio("Is the storey range High?", [0, 1], index=0)

# Prediction button
if st.button("Predict"):
    # Collect features based on user inputs
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

    # Align features to match trained model columns
    aligned_user_input = pd.DataFrame(features, columns=trained_columns)

    # Convert the features to numpy array for prediction
    user_input_array = aligned_user_input.to_numpy()

    # Use the model to make a prediction
    try:
        prediction_log = model.predict(user_input_array)
        # Exponentiate the prediction (if the model returns log-transformed price)
        prediction_actual = np.exp(prediction_log)
        st.success(f"The predicted resale price is: **${prediction_actual[0]:,.2f}**")
    except Exception as e:
        st.error(f"Error: {e}")