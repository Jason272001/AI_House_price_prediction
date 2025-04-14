import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df=pd.read_csv("kc_house_data_large.csv")

features=['sqft_living','bedrooms','bathrooms','floors','condition','grade','waterfront']

x=df[features]
y=df[['price']]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

predictions=model.predict(x_test)

mse=mean_squared_error(y_test,predictions)
r2=r2_score(y_test,predictions)

print("Mean Squred Error:",mse)
print("R2 score:",r2)

plt.scatter(x_test['sqft_living'],y_test,color='blue',label='Actual')
plt.scatter(x_test['sqft_living'],predictions,color='red',label='Predictied')
plt.xlabel('Living Area (sqft)')
plt.ylabel('Price')
plt.title('House Price Prediction (Multi-Feature)')
plt.legend()
plt.show()