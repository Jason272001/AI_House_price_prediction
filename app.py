import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained model and scaler
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load ZIP-level average price data from Zillow
zillow_df = pd.read_csv("zillow_clean.csv")  # Make sure this is the correct file name


# Page layout
st.set_page_config(page_title="Nationwide House Price Predictor", layout="centered")
st.markdown("<h1 style='text-align:center;'>🏠 House Price Predictor (ZIP Smart)</h1>", unsafe_allow_html=True)

# Input fields
sqft = st.slider("Living Area (sqft)", 500, 5000, step=50)
bedrooms = st.slider("Bedrooms", 1, 10)
bathrooms = st.slider("Bathrooms", 1, 5)
floors = st.selectbox("Floors", [1, 2, 3])
condition = st.slider("Condition (1 = Poor, 5 = Excellent)", 1, 5)
grade = st.slider("Grade (1 = Low, 13 = High)", 1, 13)
waterfront = st.selectbox("Waterfront View", [0, 1])

# ZIP Code dropdown from Zillow data
zipcodes = sorted(zillow_df['zipcode'].dropna().unique())
selected_zip = st.selectbox("Select ZIP Code", zipcodes)

# Get the average price in the selected ZIP
zip_price = zillow_df[zillow_df['zipcode'] == selected_zip]['avg_zip_price'].values[0]

# Create input dataframe
input_df = pd.DataFrame([[sqft, bedrooms, bathrooms, floors, condition, grade, waterfront, selected_zip, zip_price]],
    columns=['sqft_living','bedrooms','bathrooms','floors','condition','grade','waterfront','zipcode','avg_zip_price'])

# Predict when button is clicked
if st.button("Predict Price 💰"):
    scaled_input = scaler.transform(input_df)
    prediction = float(model.predict(scaled_input)[0])
    st.success(f"🏡 Estimated House Price: **${int(prediction):,}**")

    # Visualize as bar chart
    fig, ax = plt.subplots()
    ax.bar(["Predicted Price"], [prediction], color='red')
    st.pyplot(fig)

    # Option to download result
    result_df = input_df.copy()
    result_df["Predicted Price"] = prediction
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Prediction", data=csv, file_name="prediction.csv", mime="text/csv")
