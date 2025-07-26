import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model and scaler
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🏡 Housing Price Prediction App")
st.markdown("Predict house prices using your trained XGBoost model")

# User inputs
bedrooms = st.slider("Bedrooms", 1, 11, 3)
bathrooms = st.slider("Bathrooms", 0.5, 5.0, 2.0, step=0.25)
floors = st.slider("Floors", 1.0, 3.5, 1.0, step=0.5)
waterfront = st.selectbox("Waterfront (0 = No, 1 = Yes)", [0, 1])
view = st.selectbox("View (0 = No, 1 = Yes)", [0, 1])
condition = st.slider("Condition (1 to 5)", 1, 5, 3)
grade = st.slider("Grade (1 to 13)", 1, 13, 7)
yr_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2000)
yr_renovated = st.selectbox("Renovated? (0 = No, 1 = Yes)", [0, 1])
lat = st.number_input("Latitude", value=47.5112)
long = st.number_input("Longitude", value=-122.257)
sqft_living15 = st.number_input("Sqft Living 15", value=1500)

# Raw inputs (we'll log transform)
sqft_living = st.number_input("Sqft Living", value=1800)
sqft_above = st.number_input("Sqft Above", value=1200)
sqft_basement = st.number_input("Sqft Basement", value=600)

# Log transform
sqft_living_log = np.log1p(sqft_living)
sqft_above_log = np.log1p(sqft_above)
sqft_basement_log = np.log1p(sqft_basement)

# Dataframe for prediction
input_data = pd.DataFrame([[
    bedrooms, bathrooms, floors, waterfront, view,
    condition, grade, yr_built, yr_renovated,
    lat, long, sqft_living15,
    sqft_living_log, sqft_above_log, sqft_basement_log
]], columns=[
    'bedrooms', 'bathrooms', 'floors', 'waterfront', 'view',
    'condition', 'grade', 'yr_built', 'yr_renovated',
    'lat', 'long', 'sqft_living15',
    'sqft_living_log', 'sqft_above_log', 'sqft_basement_log'
])

# Scale the input
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict Price"):
    price_log = model.predict(input_scaled)
    price = np.expm1(price_log)  # Convert back to original price
    st.success(f"🏷️ Predicted House Price: **${price[0]:,.2f}**")
