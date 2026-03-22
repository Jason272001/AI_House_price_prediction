from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "house_price_model.pkl")


@st.cache_resource
def load_scaler():
    return joblib.load(BASE_DIR / "scaler.pkl")


@st.cache_resource
def load_zillow():
    df = pd.read_csv(BASE_DIR / "zillow_clean.csv")
    # One row per ZIP for lookups
    return df.groupby("zipcode", as_index=False)["avg_zip_price"].mean()


@st.cache_resource
def load_metrics():
    path = BASE_DIR / "model_metrics.pkl"
    if path.exists():
        return joblib.load(path)
    return None


def normalize_zip_code(raw: str):
    """Parse US ZIP (5 digits; tolerate 4-digit entries missing a leading zero)."""
    s = "".join(ch for ch in str(raw).strip() if ch.isdigit())
    if not s:
        return None
    if len(s) == 4:
        s = "0" + s
    if len(s) != 5:
        return None
    return int(s)


def lookup_zip_avg_price(zillow_df: pd.DataFrame, zip_int: int) -> Optional[float]:
    zc = zillow_df["zipcode"]
    match = zillow_df[zc.astype(int) == zip_int]
    if match.empty:
        return None
    return float(match["avg_zip_price"].iloc[0])


def require_artifacts():
    missing = []
    for name in ("house_price_model.pkl", "scaler.pkl"):
        if not (BASE_DIR / name).exists():
            missing.append(name)
    if missing:
        st.error(
            "Missing model files: "
            + ", ".join(missing)
            + ". Run `python model.py` from the project folder to train and generate them."
        )
        st.stop()


require_artifacts()

model = load_model()
scaler = load_scaler()
zillow_df = load_zillow()
metrics = load_metrics()

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.markdown(
    "<h1 style='text-align:center;'>House price estimate by home & ZIP market</h1>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("### About")
st.sidebar.info(
    "This tool blends **home features** (size, beds/baths, quality) with a **ZIP-level "
    "average price** from your data file. It is **not** an appraisal or lending estimate—"
    "use it for exploration only."
)
if metrics:
    st.sidebar.caption(f"Hold-out R² (training run): {metrics['r2']:.3f}")

sqft = st.slider("Living area (sq ft)", 500, 5000, value=2000, step=50)
bedrooms = st.slider("Bedrooms", 1, 10, value=3)
bathrooms = st.slider("Bathrooms", 0.5, 6.0, value=2.0, step=0.5)
floors = st.selectbox("Floors", [1, 2, 3], index=1)
condition = st.slider("Condition (1 = poor, 5 = excellent)", 1, 5, value=3)
grade = st.slider("Grade (1 = low, 13 = high)", 1, 13, value=7)
waterfront = st.selectbox("Waterfront view", [0, 1], format_func=lambda x: "Yes" if x else "No")

zip_raw = st.text_input(
    "ZIP code (US, 5 digits)",
    value="98101",
    help="Enter a ZIP from your Zillow file. Leading zeros optional (e.g. 0701 → 00701).",
)
zip_norm = normalize_zip_code(zip_raw)
zip_price = None
if zip_norm is not None:
    zip_price = lookup_zip_avg_price(zillow_df, zip_norm)

if zip_norm is None:
    st.warning("Enter a valid 5-digit US ZIP (or 4 digits with a leading zero implied).")
elif zip_price is None:
    st.warning(f"No average price found for ZIP **{zip_norm:05d}** in the loaded data.")

if st.button("Estimate price", type="primary"):
    if zip_price is None:
        st.error("Fix the ZIP code above before estimating.")
        st.stop()

    input_df = pd.DataFrame(
        [
            [
                sqft,
                bedrooms,
                bathrooms,
                float(floors),
                condition,
                grade,
                waterfront,
                zip_price,
            ]
        ],
        columns=[
            "sqft_living",
            "bedrooms",
            "bathrooms",
            "floors",
            "condition",
            "grade",
            "waterfront",
            "avg_zip_price",
        ],
    )

    scaled_input = scaler.transform(input_df)
    raw = model.predict(scaled_input)
    # sklearn may return (n,), (n,1), or a 0-d array; avoid [0] on a bare float
    prediction = float(np.asarray(raw, dtype=np.float64).ravel()[0])
    prediction = max(prediction, 0.0)

    st.success(f"Estimated price: **${prediction:,.0f}**")
    st.caption(
        f"ZIP market average in data: **${zip_price:,.0f}** — compare to see if the estimate "
        "sits above or below typical for that area."
    )

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(["Your estimate"], [prediction], color="#c0392b")
    ax.set_ylabel("USD")
    ax.yaxis.set_major_formatter(lambda x, pos: f"${x:,.0f}")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    result_df = input_df.copy()
    result_df["zipcode"] = zip_norm
    result_df["predicted_price"] = prediction
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download result (CSV)",
        data=csv,
        file_name="house_price_estimate.csv",
        mime="text/csv",
    )
