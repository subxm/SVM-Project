import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Jordan SVM Profitability Predictor", page_icon="👟", layout="centered")

st.title("Jordan SVM Profitability Predictor")
st.write("Predict whether a sneaker transaction is likely to be profitable.")

# Load artifacts
model = joblib.load("artifacts/svm_profitability_pipeline.joblib")
with open("artifacts/model_metadata.json", "r") as f:
    metadata = json.load(f)

categorical_cols = metadata["categorical_cols"]
numeric_cols = metadata["numeric_cols"]

# Input widgets
st.subheader("Enter Transaction Features")

# Categorical options from original data
df_ref = pd.read_csv("jordan_market_dataset_2026.csv")

shoe_model = st.selectbox("Shoe Model", sorted(df_ref["Shoe_Model"].dropna().unique()))
colorway = st.selectbox("Colorway", sorted(df_ref["Colorway"].dropna().unique()))
condition = st.selectbox("Condition", sorted(df_ref["Condition"].dropna().unique()))
sales_channel = st.selectbox("Sales Channel", sorted(df_ref["Sales_Channel"].dropna().unique()))

retail_price = st.number_input("Retail Price (USD)", min_value=0.0, value=200.0, step=1.0)
days_inventory = st.number_input("Days in Inventory", min_value=0, value=20, step=1)

threshold = st.slider("Decision Threshold", min_value=0.30, max_value=0.90, value=0.50, step=0.01)
st.caption("If results look too strict, try a threshold around 0.45 to 0.55.")

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "Shoe_Model": shoe_model,
        "Colorway": colorway,
        "Condition": condition,
        "Sales_Channel": sales_channel,
        "Retail_Price_USD": retail_price,
        "Days_in_Inventory": days_inventory
    }])

    # Ensure column order matches training metadata
    input_data = input_data[metadata["feature_columns"]]

    prob_profitable = model.predict_proba(input_data)[0, 1]
    model_default_pred = int(model.predict(input_data)[0])
    threshold_pred = int(prob_profitable >= threshold)

    st.subheader("Prediction Result")
    st.metric("Probability of Profitability", f"{prob_profitable:.2%}")
    st.write(f"Model default decision (SVM): {'Likely Profitable' if model_default_pred == 1 else 'Likely Not Profitable'}")
    st.write(f"Threshold-based decision (@ {threshold:.2f}): {'Likely Profitable' if threshold_pred == 1 else 'Likely Not Profitable'}")

    if threshold_pred == 1:
        st.success("Likely Profitable")
    else:
        st.error("Likely Not Profitable")