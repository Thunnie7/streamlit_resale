import streamlit as st
import joblib

model = joblib.load('GBR_model.pkl')

col1, col2, col3 = st.columns(3)
with col1:
    floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=1, max_value=500)
    house_age = st.number_input("House Age", min_value=0, max_value=99)

with col2:
    year = st.number_input("Year", min_value=2000, max_value=2030)
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

    try:
        prediction = model.predict(features)
        st.success(f"The predicted resale price is: **${prediction[0]:,.2f}**")
    except Exception as e:
        st.error(f"Error: {e}")

