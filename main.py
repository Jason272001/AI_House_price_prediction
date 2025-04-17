import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib

df=pd.read_csv("kc_house_data_large.csv")

features=['sqft_living','bedrooms','bathrooms','floors','condition','grade','waterfront']

x=df[features]
y=df[['price']]

x_unscaled=x.copy()

plt.figure(figsize=(10,6))
sns.heatmap(df[features+['price']].corr(),annot=True,cmap='coolwarm')
plt.title("Feature Correlation with Price")
plt.show()

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
joblib.dump(scaler, "scaler.pkl")

x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=1)
model=LinearRegression()
model.fit(x_train,y_train)

predictions=model.predict(x_test)

mse=mean_squared_error(y_test,predictions)
r2=r2_score(y_test,predictions)

print("Mean Squred Error:",mse)
print("R2 score:",r2)

plt.figure(figsize=(10, 6))
plt.scatter(x_unscaled.iloc[y_test.index]['sqft_living'], y_test, color='blue', label='Actual')
plt.scatter(x_unscaled.iloc[y_test.index]['sqft_living'], predictions, color='red', label='Predicted')
plt.xlabel('Living Area (sqft)')
plt.ylabel('Price')
plt.title('House Price Prediction (Multi-Feature)')
plt.legend()
plt.show()


joblib.dump(model,"house_price_model.pkl")
print("Model Saved Successfully")