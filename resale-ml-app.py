import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained Gradient Boosting Regressor model
model = joblib.load('GBR_model.pkl')

# Define the function to predict resale price
def predict_resale_price(floor_area_sqm, house_age, year, block_numeric, flat_model_encoded, town_encoded,
                         flat_type_2_room, flat_type_3_room, flat_type_4_room, flat_type_5_room, flat_type_executive,
                         flat_type_multi_generation, storey_range_binned_low, storey_range_binned_medium, storey_range_binned_high):
    # Create a dictionary with the input features
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

    # Convert input data into a DataFrame
    features = pd.DataFrame(data, index=[0])

    # Convert the input features into numpy array for prediction
    user_input_array = np.array(features).reshape(1, -1)

    # Ensure that the input array has the correct shape
    print("Shape of user input array:", user_input_array.shape)

    # Use the model to predict the log price
    predicted_log_price = model.predict(user_input_array)  # This is the log prediction

    # Exponentiate the log prediction to get the actual price
    predicted_actual_price = np.exp(predicted_log_price)

    return predicted_log_price[0], predicted_actual_price[0]

# Example inputs
predicted_log_price, predicted_actual_price = predict_resale_price(
    floor_area_sqm=44.0,
    house_age=37.666667,
    year=2017,
    block_numeric=406,
    flat_model_encoded=5,
    town_encoded=0,
    flat_type_2_room=1, flat_type_3_room=0, flat_type_4_room=0,
    flat_type_5_room=0, flat_type_executive=0, flat_type_multi_generation=0,
    storey_range_binned_low=0, storey_range_binned_medium=1, storey_range_binned_high=0
)

print(f"Predicted Log Resale Price: {predicted_log_price}")
print(f"Predicted Actual Resale Price: {predicted_actual_price}")
