import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("GBR_model.pkl")

# Title of the app
st.title("Resale Price Prediction App")
st.write("Predict resale prices based on input features!")

# Input fields for the model
floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=10.0, max_value=200.0, step=1.0)
house_age = st.number_input("House Age (Years)", min_value=0.0, max_value=99.0, step=1.0)
year = st.number_input("Year of Sale", min_value=2000, max_value=2025, step=1)
block_numeric = st.number_input("Block Number (Numeric)", min_value=1, max_value=999, step=1)
flat_model_encoded = st.number_input("Flat Model (Encoded)", min_value=0, max_value=10, step=1)
town_encoded = st.number_input("Town (Encoded)", min_value=0, max_value=30, step=1)

# One-hot encoded flat type
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

# Create a predict button
if st.button("Predict Resale Price"):
    # Prepare input features
    input_features = [
        floor_area_sqm, house_age, year, block_numeric, flat_model_encoded, town_encoded,
        flat_type_2_room, flat_type_3_room, flat_type_4_room, flat_type_5_room,
        flat_type_executive, flat_type_multi_generation,
        storey_range_binned_low, storey_range_binned_medium, storey_range_binned_high
    ]
    input_array = np.array(input_features).reshape(1, -1)

    # Predict resale price
    predicted_price = model.predict(input_array)

    # Display the prediction
    st.success(f"The predicted resale price is: ${predicted_price[0]:,.2f}")
