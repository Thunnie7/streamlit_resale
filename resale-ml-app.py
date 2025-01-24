import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('GBR_model.pkl')

# Dropdown options for flat model and town (replace with your actual encoded values)
flat_model_options = [0, 1, 2, 3, 4, 5]  # Replace with actual encoded values
town_options = [0, 1, 2, 3, 4, 5, 6, 7]  # Replace with actual encoded values

# Define a function for predictions
def predict_resale_price(
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
):
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

    # Make predictions
    predicted_log_price = model.predict(features)[0]

    # Convert log-predicted price back to actual price (if necessary)
    predicted_actual_price = np.exp(predicted_log_price)

    return predicted_log_price, predicted_actual_price


# Streamlit app layout for user input
st.title("🏠 House Resale Price Prediction")
st.markdown("This app predicts the resale price of a house based on various input features. Fill in the form below and hit **Predict**!")

col1, col2, col3 = st.columns(3)

with col1:
    floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=1, max_value=500, value=50)
    house_age = st.number_input("House Age", min_value=0, max_value=99, value=30)

with col2:
    year = st.number_input("Year of Sale", min_value=2000, max_value=2025, value=2020)
    block_numeric = st.number_input("Block Number (Numeric)", min_value=1, max_value=999, value=100)

with col3:
    flat_model_encoded = st.selectbox("Select Flat Model (Encoded)", flat_model_options)
    town_encoded = st.selectbox("Select Town (Encoded)", town_options)

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
    try:
        # Call the prediction function
        predicted_log_price, predicted_actual_price = predict_resale_price(
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
        )

        # Display the predictions
        
        st.success(f"Predicted Actual Resale Price: **${predicted_actual_price:,.2f}**")
    except Exception as e:
        st.error(f"Error: {e}")
