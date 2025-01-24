import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor  
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.write("""
# Custom Dataset Prediction App
This app predicts the **target variable** based on input features!
""")

st.sidebar.header('User Input Parameters')

# Define the input features for your dataset
def user_input_features():
    floor_area_sqm = st.sidebar.slider('Floor Area (sqm)', 10.0, 200.0, 50.0)
    house_age = st.sidebar.slider('House Age (Years)', 0, 99, 30)
    year = st.sidebar.slider('Year of Sale', 2000, 2025, 2020)
    block_numeric = st.sidebar.slider('Block Number (Numeric)', 1, 999, 100)
    flat_model_encoded = st.sidebar.slider('Flat Model (Encoded)', 0, 10, 2)
    town_encoded = st.sidebar.slider('Town (Encoded)', 0, 30, 15)
    
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

    data = {
        'Floor Area (sqm)': floor_area_sqm,
        'House Age (Years)': house_age,
        'Year of Sale': year,
        'Block Number (Numeric)': block_numeric,
        'Flat Model (Encoded)': flat_model_encoded,
        'Town (Encoded)': town_encoded,
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

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

# Simulating a dataset for this example (replace this with your actual data)
# Example dataset with appropriate features
np.random.seed(42)
X = np.random.rand(1000, 15)  # Replace this with your actual feature matrix (15 features as per the input)
y = np.random.rand(1000)      # Replace with your actual target variable (e.g., resale price)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Gradient Boosting Regressor model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
prediction = model.predict(df)

st.subheader('Prediction')
st.write(f"The predicted value is: **{prediction[0]:.2f}**")

# Evaluate model performance (optional, if needed for testing)
y_test_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
st.write(f"Model Mean Squared Error (MSE) on test data: **{mse:.2f}**")
