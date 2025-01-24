import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Load the pre-trained Gradient Boosting Regressor model
model = joblib.load('GBR_model.pkl')

# Load the dataset from CSV
df = pd.read_csv("ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv")

st.write("""
# Custom Dataset Prediction App
This app predicts the **target variable** based on input features!
""")

st.sidebar.header('User Input Parameters')

# Define the input features for your dataset using text input
def user_input_features():
    floor_area_sqm = st.sidebar.text_input('Floor Area (sqm)', '50.0')
    house_age = st.sidebar.text_input('House Age (Years)', '30')
    year = st.sidebar.text_input('Year of Sale', '2020')
    block_numeric = st.sidebar.text_input('Block Number (Numeric)', '100')
    flat_model_encoded = st.sidebar.text_input('Flat Model (Encoded)', '2')
    town_encoded = st.sidebar.text_input('Town (Encoded)', '15')

    # One-hot encoded flat type
    flat_type_2_room = st.sidebar.radio("Is it a 2 ROOM?", [0, 1])
    flat_type_3_room = st.sidebar.radio("Is it a 3 ROOM?", [0, 1])
    flat_type_4_room = st.sidebar.radio("Is it a 4 ROOM?", [0, 1])
    flat_type_5_room = st.sidebar.radio("Is it a 5 ROOM?", [0, 1])
    flat_type_executive = st.sidebar.radio("Is it an EXECUTIVE?", [0, 1])
    flat_type_multi_generation = st.sidebar.radio("Is it a MULTI-GENERATION?", [0, 1])

    # One-hot encoded storey range
    storey_range_binned_low = st.sidebar.radio("Is the storey range Low?", [0, 1])
    storey_range_binned_medium = st.sidebar.radio("Is the storey range Medium?", [0, 1])
    storey_range_binned_high = st.sidebar.radio("Is the storey range High?", [0, 1])

    # Convert text input to float and int for computation
    data = {
        'Floor Area (sqm)': float(floor_area_sqm),
        'House Age (Years)': float(house_age),
        'Year of Sale': int(year),
        'Block Number (Numeric)': int(block_numeric),
        'Flat Model (Encoded)': int(flat_model_encoded),
        'Town (Encoded)': int(town_encoded),
        'Flat Type 2 Room': flat_type_2_room,
        'Flat Type 3 Room': flat_type_3_room,
        'Flat Type 4 Room': flat_type_4_room,
        'Flat Type 5 Room': flat_type_5_room,
        'Flat Type Executive': flat_type_executive,
        'Flat Type Multi Generation': flat_type_multi_generation,
        'Storey Range Low': storey_range_binned_low,
        'Storey Range Medium': storey_range_binned_medium,
        'Storey Range High': storey_range_binned_high
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input features
user_features = user_input_features()

st.subheader('User Input Parameters')
st.write(user_features)

# Use the loaded model to make a prediction
prediction = model.predict(user_features)

st.subheader('Prediction')
st.write(f"The predicted value is: **{prediction[0]:.2f}**")

# Optional: Evaluate model performance with a test dataset
# Note: If you want to perform this, make sure to define X_train, y_train, X_test, and y_test from your actual dataset
X = df.drop('resale_price', axis=1)  # Replace with the actual feature columns
y = df['resale_price']  # Replace with the actual target column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_test_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)

st.write(f"Model Mean Squared Error (MSE) on test data: **{mse:.2f}**")
