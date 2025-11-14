# app_fixed.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from pathlib import Path

st.set_page_config(page_title="Workers Compensation Claim Cost Predictor")

st.title("Workers Compensation Claim Cost Predictor")

# Load preprocessors + model (ensure relative path from app)
MODELS_DIR = Path("./models")
scaler = joblib.load(MODELS_DIR / "robust_scaler.joblib")
ohe = joblib.load(MODELS_DIR / "ohe.joblib")
nn_model = tf.keras.models.load_model(MODELS_DIR / "nn_model.keras")

# --- User inputs -------------------------------------------------------------
st.header("Enter Claim Details")

Age = st.number_input("Age", min_value=16, max_value=100, value=30)
WeeklyPay = st.number_input("Weekly Pay ($)", min_value=0.0, value=800.0)
HoursWorkedPerWeek = st.number_input("Hours Worked Per Week", min_value=1.0, value=40.0)
InitialCaseEstimate = st.number_input("Initial Case Estimate ($)", min_value=0.0, value=1000.0)
report_delay_days = st.number_input("Report Delay (days)", min_value=0, value=2)
accident_hour = st.number_input("Accident Hour (0-23)", min_value=0, max_value=23, value=14)

Gender = st.selectbox("Gender", ["M","F"])
MaritalStatus = st.selectbox("Marital Status", ["M","S","U"])
DependentChildren = st.selectbox("Dependent Children", ["0","1","2","3","4","5","6"])
DependentsOther = st.selectbox("Other Dependents", ["0","1","2"])
DaysWorkedPerWeek = st.selectbox("Days Worked Per Week", ["1","2","3","4","5","6","7"])
PartTimeFullTime = st.selectbox("Employment Type", ["F","P"])
accident_weekday = st.selectbox("Accident Weekday", 
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
accident_month = st.selectbox("Accident Month", 
    ["January","February","March","April","May","June","July","August","September","October","November","December"])

if st.button("Predict Ultimate Claim Cost"):

    # --------------------- ENGINEERING FEATURES (RAW) --------------------------
    # Derived features computed from raw input values (match training)
    HourlyWage_raw = WeeklyPay / (HoursWorkedPerWeek if HoursWorkedPerWeek != 0 else 0.001)
    Pay_x_Hours_raw = WeeklyPay * HoursWorkedPerWeek
    is_weekend_flag = 1 if accident_weekday in ["Saturday","Sunday"] else 0

    # Build single-row DataFrame containing raw columns and categorical columns
    df = pd.DataFrame([{
        "Age": Age,
        "WeeklyPay_raw": WeeklyPay,
        "HoursWorkedPerWeek_raw": HoursWorkedPerWeek,
        "InitialCaseEstimate_raw": InitialCaseEstimate,
        "report_delay_days": report_delay_days,
        "accident_hour": accident_hour,
        "HourlyWage_raw": HourlyWage_raw,
        "Pay_x_Hours_raw": Pay_x_Hours_raw,
        "is_weekend": is_weekend_flag,
        "Gender": Gender,
        "MaritalStatus": MaritalStatus,
        "DependentChildren": DependentChildren,
        "DependentsOther": DependentsOther,
        "DaysWorkedPerWeek": DaysWorkedPerWeek,
        "PartTimeFullTime": PartTimeFullTime,
        "accident_weekday": accident_weekday,
        "accident_month": accident_month
    }])

    # --------------------- MAP raw -> training column NAMES and LOG-TRANSFORM -------------------
    # IMPORTANT: these exact target column names must match those used when scaler was fitted.
    df["WeeklyPay"] = np.log1p(df["WeeklyPay_raw"].clip(lower=0))
    df["HoursWorkedPerWeek"] = np.log1p(df["HoursWorkedPerWeek_raw"].clip(lower=0))
    df["InitialCaseEstimate"] = np.log1p(df["InitialCaseEstimate_raw"].clip(lower=0))
    df["HourlyWage"] = np.log1p(df["HourlyWage_raw"].clip(lower=0))
    df["Pay_x_Hours"] = np.log1p(df["Pay_x_Hours_raw"].clip(lower=0))

    # Drop the _raw columns now that we've created the exact names used during training
    df = df.drop(columns=[
        "WeeklyPay_raw","HoursWorkedPerWeek_raw","InitialCaseEstimate_raw",
        "HourlyWage_raw","Pay_x_Hours_raw"
    ])

    # --------------------- NUMERIC + CATEGORICAL COLUMN ORDER (MUST MATCH TRAINING) ----------
    # These numeric column names must match exactly the DataFrame columns used when the scaler was fitted.
    numeric_cols = [
        "Age", "WeeklyPay", "HoursWorkedPerWeek", "InitialCaseEstimate",
        "report_delay_days", "accident_hour", "HourlyWage", "Pay_x_Hours"
    ]

    # Categorical columns include is_weekend (note: encoder was fitted on strings)
    categorical_cols = [
        "Gender","MaritalStatus","DependentChildren","DependentsOther",
        "DaysWorkedPerWeek","PartTimeFullTime","accident_weekday","accident_month","is_weekend"
    ]

    # Ensure is_weekend is string (encoder was fitted on stringified categories)
    df["is_weekend"] = df["is_weekend"].astype(str)

    # Also ensure all categorical cols are strings
    for c in categorical_cols:
        if c != "is_weekend":  # already cast above
            df[c] = df[c].astype(str)

    # --------------------- APPLY SCALER + OHE (IN SAME ORDER AS TRAINING) --------------------
    try:
        X_num_scaled = scaler.transform(df[numeric_cols])
    except ValueError as e:
        st.error("Scaler transform failed: " + str(e))
        st.stop()

    try:
        X_cat = ohe.transform(df[categorical_cols])
    except Exception as e:
        st.error("OneHotEncoder transform failed: " + str(e))
        st.stop()

    # FINAL feature array: scaled numerics followed by OHE categorical (same concatenation used in training)
    X_final = np.concatenate([X_num_scaled, X_cat], axis=1)

    # --------------------- PREDICT ----------------------------
    y_pred_log = nn_model.predict(X_final)
    # y_pred_log shape: (1,1) or (1,) depending on TF version - handle both
    if np.ndim(y_pred_log) == 2:
        y_pred_log_value = float(y_pred_log[0,0])
    else:
        y_pred_log_value = float(y_pred_log[0])

    y_pred_dollars = np.expm1(y_pred_log_value)

    st.subheader("Predicted Ultimate Incurred Claim Cost")
    st.success(f"${y_pred_dollars:,.2f}")

    # optional: show intermediate final feature vector length
    st.write(f"Model input vector length: {X_final.shape[1]}")
