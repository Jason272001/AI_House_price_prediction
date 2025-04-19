import streamlit as st
import pandas as pd
import joblib

scaler = joblib.load("scaler.pkl")

model = joblib.load("house_price_model.pkl")
st.set_page_config(page_title="House Price Predictior",layout="centered")
st.title("House Price Predictor")
st.markdown("Enter the house details below to predict the estimated price")
st.markdown("### 💡 How It Works")
st.write("This app uses a trained Linear Regression model to predict house prices based on...")

sqft=st.slider("living Area(sqft)",500,5000,step=50)
bedrooms=st.slider("Bedrooms",1,6)
bathrooms=st.slider("Bathrooms",1,4)
floors=st.selectbox("Floors",[1,2])
condition=st.slider("condition(1=Poor,5=Excellent)",1,5)
grade=st.slider("Grade(1=low,13=Hight)",1,13)
waterfront = st.selectbox("Waterfront View", [0, 1])

input_df=pd.DataFrame([[sqft,bedrooms,bathrooms,floors,condition,grade,waterfront]],
                     columns=['sqft_living','bedrooms','bathrooms','floors','condition','grade','waterfront'] )
# Scale input before prediction


if st.button("predict Price "):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    st.success(f"Estmited House price : ${int(prediction):,}")

