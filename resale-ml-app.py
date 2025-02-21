import streamlit as st
import joblib
import numpy as np
import pandas as pd  

# Load the trained model
try:
    model = joblib.load('GBR_model.pkl')
    st.write("✅ Model Loaded Successfully!")
except Exception as e:
    model = None
    st.error(f"❌ Model Loading Error: {e}")

# Dropdown options for flat model and town (replace with your actual encoded values)
flat_model_options = [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] 
town_options = [0, 1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25] 

# Define feature column names (must match training data)
feature_columns = [
    "floor_area_sqm", "house_age", "year", "block_numeric", "flat_model_encoded", "town_encoded",
    "flat_type_2_room", "flat_type_3_room", "flat_type_4_room", "flat_type_5_room",
    "flat_type_executive", "flat_type_multi_generation", "storey_range_binned_low",
    "storey_range_binned_medium", "storey_range_binned_high"
]

# Streamlit app layout for user input
st.title("🏠 House Resale Price Prediction")
st.markdown("This app predicts the resale price of a house based on various input features. Fill in the form below and hit **Predict**!")

col1, col2, col3 = st.columns(3)

with col1:
    floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=1, max_value=500, value=50)
    house_age = st.number_input("House Age (Years)", min_value=0, max_value=99, value=30)

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

if st.button("Predict"):
    try:
        # Ensure model is loaded
        if model is None:
            st.error("❌ Model not loaded correctly. Check the model file.")
        else:
            # Convert categorical values to indices
            flat_model_encoded = flat_model_options.index(flat_model_encoded) if flat_model_encoded in flat_model_options else 0
            town_encoded = town_options.index(town_encoded) if town_encoded in town_options else 0

            # Collect features into an array
            features = [[
                floor_area_sqm, house_age, year, block_numeric, flat_model_encoded, town_encoded,
                flat_type_2_room, flat_type_3_room, flat_type_4_room, flat_type_5_room,
                flat_type_executive, flat_type_multi_generation, storey_range_binned_low,
                storey_range_binned_medium, storey_range_binned_high
            ]]

            # ✅ Create DataFrame using the correct feature names
            aligned_user_input = pd.DataFrame(features, columns=feature_columns)

            # Debug: Ensure feature alignment
            st.write("✅ Model Expected Columns:", feature_columns)
            st.write("📌 User Input Columns:", aligned_user_input.columns.tolist())

            # Convert to NumPy array
            user_input_array = np.array(aligned_user_input, dtype=float).reshape(1, -1)

            # Debugging Output
            st.write("📏 Final Input Shape:", user_input_array.shape)

            # Ensure input shape matches trained data
            if user_input_array.shape[1] != len(feature_columns):
                st.error(f"🚨 Shape mismatch! Expected {len(feature_columns)}, got {user_input_array.shape[1]}")
            else:
                # Make prediction
                prediction_log = model.predict(user_input_array)

                # Convert log prediction to actual price
                prediction_actual = np.exp(prediction_log)

                # Display result
                st.success(f"📈 Predicted Log Resale Price: **${prediction_log[0]:,.2f}**")
                st.success(f"💰 Predicted Resale Price: **${prediction_actual[0]:,.2f}**")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
