"""
Train a linear regression model for house price prediction.

Training data has no real ZIPs; we assign each row a market-level avg_zip_price by
matching house price rank to ZIP median price rank (quantile alignment). Raw ZIP
codes are not used as model inputs (they are identifiers, not magnitudes).
"""
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def assign_avg_zip_price_by_quantile(house_df: pd.DataFrame, zillow_df: pd.DataFrame) -> pd.DataFrame:
    """Align each home's price rank with a ZIP market's avg price rank."""
    z = (
        zillow_df.groupby("zipcode", as_index=False)["avg_zip_price"]
        .mean()
        .sort_values("avg_zip_price")
        .reset_index(drop=True)
    )
    h = house_df.sort_values("price").reset_index(drop=True)
    n, m = len(h), len(z)
    if m < 2:
        raise ValueError("Zillow data needs at least 2 ZIP rows.")
    ranks = np.linspace(0, m - 1, n, dtype=float)
    idx = np.clip(np.round(ranks).astype(int), 0, m - 1)
    out = h.copy()
    out["avg_zip_price"] = z.iloc[idx]["avg_zip_price"].values
    return out


def main():
    house_df = pd.read_csv(os.path.join(BASE_DIR, "kc_house_data_large.csv"))
    zillow_df = pd.read_csv(os.path.join(BASE_DIR, "zillow_clean.csv"))

    house_df = assign_avg_zip_price_by_quantile(house_df, zillow_df)

    features = [
        "sqft_living",
        "bedrooms",
        "bathrooms",
        "floors",
        "condition",
        "grade",
        "waterfront",
        "avg_zip_price",
    ]
    x = house_df[features]
    y = house_df[["price"]]
    x_unscaled = x.copy()

    plt.figure(figsize=(10, 6))
    sns.heatmap(house_df[features + ["price"]].corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation with Price")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "correlation_heatmap.png"), dpi=120)
    plt.close()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.2, random_state=1
    )
    model = LinearRegression()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Mean Squared Error:", mse)
    print("R2 Score:", r2)

    plt.figure(figsize=(10, 6))
    test_idx = y_test.index
    plt.scatter(x_unscaled.loc[test_idx, "sqft_living"], y_test, color="blue", label="Actual")
    plt.scatter(x_unscaled.loc[test_idx, "sqft_living"], predictions, color="red", label="Predicted")
    plt.xlabel("Living Area (sqft)")
    plt.ylabel("Price")
    plt.title("House Price Prediction (Multi-Feature)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "predictions_vs_actual.png"), dpi=120)
    plt.close()

    metrics = {"mse": float(mse), "r2": float(r2)}
    joblib.dump(model, os.path.join(BASE_DIR, "house_price_model.pkl"))
    joblib.dump(metrics, os.path.join(BASE_DIR, "model_metrics.pkl"))
    print("Model and metrics saved successfully.")


if __name__ == "__main__":
    main()
