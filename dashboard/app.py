
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_assets():
    model = joblib.load(
        BASE_DIR / "models" / "smartshield_model.joblib"
    )

    scaler = joblib.load(
        BASE_DIR / "models" / "smartshield_scaler.joblib"
    )

    with open(BASE_DIR / "config.json") as file:
        config = json.load(file)

    return model, scaler, config


model, scaler, config = load_assets()

st.set_page_config(
    page_title="SmartShield Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ SmartShield")
st.subheader(
    "Credit Card Fraud Detection and Risk Assessment System"
)

st.info(
    "Upload a CSV containing the 30 transaction input features. "
    "The optional Class column is ignored during prediction."
)

st.sidebar.header("Model Information")
st.sidebar.write("Selected model:", config["model_name"])
st.sidebar.write(
    "Optimized threshold:",
    round(config["threshold"], 4)
)

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)

if uploaded_file is not None:

    raw = pd.read_csv(uploaded_file)

    features = config["features"]
    scale_columns = config["scale_columns"]

    missing = [
        column for column in features
        if column not in raw.columns
    ]

    if missing:
        st.error(
            "Missing required columns: " + ", ".join(missing)
        )

    else:
        X = raw[features].copy()

        for column in features:
            X[column] = pd.to_numeric(
                X[column],
                errors="coerce"
            )

        if X.isnull().any().any():
            st.error(
                "Input contains missing or non-numeric feature values."
            )

        else:
            X.loc[:, scale_columns] = scaler.transform(
                X[scale_columns]
            )

            probabilities = model.predict_proba(X)[:, 1]

            threshold = config["threshold"]

            predictions = (
                probabilities >= threshold
            ).astype(int)

            risk_scores = probabilities * 100

            risk_levels = np.select(
                [
                    risk_scores < 30,
                    risk_scores < 70
                ],
                ["Low", "Medium"],
                default="High"
            )

            results = pd.DataFrame({
                "Fraud Probability (%)": np.round(
                    risk_scores, 2
                ),
                "Risk Score": np.round(risk_scores, 2),
                "Risk Level": risk_levels,
                "Prediction": np.where(
                    predictions == 1,
                    "FRAUD",
                    "LEGITIMATE"
                )
            })

            if "Amount" in raw.columns:
                results.insert(
                    0,
                    "Amount",
                    raw["Amount"].values
                )

            total = len(results)
            fraud_count = int((predictions == 1).sum())
            legitimate_count = total - fraud_count

            st.divider()

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Transactions", total)
            col2.metric("Predicted Fraud", fraud_count)
            col3.metric("Predicted Legitimate", legitimate_count)

            st.subheader("Transaction Predictions")

            st.dataframe(
                results,
                use_container_width=True
            )

            csv_data = results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download Prediction Results",
                data=csv_data,
                file_name="smartshield_predictions.csv",
                mime="text/csv"
            )

st.divider()

st.caption(
    "SmartShield is an academic ML prototype using an anonymized "
    "historical dataset. It is not a production banking system."
)
