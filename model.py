import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib

# Load your house dataset
house_df = pd.read_csv("kc_house_data_large.csv")

# Load Zillow ZIP-level data
zillow_df = pd.read_csv("ultimate.csv")
zillow_df = zillow_df.rename(columns={"RegionName": "zipcode", "2025-03-31": "avg_zip_price"})

# ✅ Inject random ZIP codes from Zillow into house_df (temporary fix)
house_df['zipcode'] = np.random.choice(zillow_df['zipcode'].dropna().unique(), size=len(house_df))

# Merge Zillow ZIP-level price into house_df
house_df = pd.merge(house_df, zillow_df[["zipcode", "avg_zip_price"]], on="zipcode", how="left")

# Define features
features = ['sqft_living', 'bedrooms', 'bathrooms', 'floors', 'condition',
            'grade', 'waterfront', 'zipcode', 'avg_zip_price']
x = house_df[features]
y = house_df[['price']]
x_unscaled = x.copy()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(house_df[features + ['price']].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation with Price")
plt.show()

# Scale features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
joblib.dump(scaler, "scaler.pkl")

# Train/test split and training
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=1)
model = LinearRegression()
model.fit(x_train, y_train)

# Prediction and evaluation
predictions = model.predict(x_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Plot predictions vs. actual
plt.figure(figsize=(10, 6))
plt.scatter(x_unscaled.iloc[y_test.index]['sqft_living'], y_test, color='blue', label='Actual')
plt.scatter(x_unscaled.iloc[y_test.index]['sqft_living'], predictions, color='red', label='Predicted')
plt.xlabel('Living Area (sqft)')
plt.ylabel('Price')
plt.title('House Price Prediction (Multi-Feature)')
plt.legend()
plt.show()

# Save model
joblib.dump(model, "house_price_model.pkl")
print("Model Saved Successfully")
