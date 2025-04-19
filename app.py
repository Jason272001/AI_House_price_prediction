import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

scaler = joblib.load("scaler.pkl")

model = joblib.load("house_price_model.pkl")
zillow_df = pd.read_csv("ultimate.csv")  # 
zillow_df = zillow_df.rename(columns={"RegionName": "zipcode", "2025-03-31": "avg_zip_price"})

st.set_page_config(page_title="House Price Predictior",layout="centered")
st.title("House Price Predictor")
st.markdown("Enter the house details below to predict the estimated price")
st.markdown("### 💡 How It Works")
st.write("This app uses a trained Linear Regression model to predict house prices based on...")





sqft=st.slider("living Area(sqft)",500,10000,step=50)
bedrooms=st.slider("Bedrooms",1,7)
bathrooms=st.slider("Bathrooms",1,6)
floors=st.selectbox("Floors",[1,2,3])
condition=st.slider("condition(1=Poor,5=Excellent)",1,5)
grade=st.slider("Grade(1=low,13=Hight)",1,13)
waterfront = st.selectbox("Waterfront View", [0, 1])
zipcodes = sorted(zillow_df['zipcode'].unique())
selected_zip = st.selectbox("Select ZIP Code", zipcodes)

# Get avg ZIP price
zip_price = zillow_df[zillow_df['zipcode'] == selected_zip]['avg_zip_price'].values[0]

# Create input DataFrame
input_df = pd.DataFrame([[sqft, bedrooms, bathrooms, floors, condition, grade, waterfront, selected_zip, zip_price]],
    columns=['sqft_living','bedrooms','bathrooms','floors','condition','grade','waterfront','zipcode','avg_zip_price'])


if st.button("predict Price "):
    scaled_input = scaler.transform(input_df)
    prediction = float(model.predict(scaled_input)[0])
    st.success(f"Estmited House price : ${int(prediction):,}")
    fig, ax = plt.subplots()

    actual_price=prediction*0.95
    ax.bar(["Acctual (Estimated)"],[actual_price],color='blue',label='Actual')
    ax.bar(["Predicted Price"], [prediction], color='red',label='Predicted')
    ax.set_ylabel("Price($)")
    ax.set_title("Predicted vs Actual Price Estimate")
    ax.legend()
    st.pyplot(fig)
    result_df=input_df.copy()
    result_df["Predicted Price"]=prediction
    csv=result_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predication",data=csv,file_name="prediction.csv",mime="text/csv")