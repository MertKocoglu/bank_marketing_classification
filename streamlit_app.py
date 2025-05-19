import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and tools
model = joblib.load("models\optimized_rf_model.pkl")
scaler = joblib.load("models\scaler.pkl")
input_columns = joblib.load("models\input_columns.pkl")

# Threshold from training (adjusted manually or via PR-curve)
best_threshold = 0.32

# Hardcoded class options instead of LabelEncoder (based on training data)
job_options = ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed"]
marital_options = ["divorced", "married", "single"]
education_options = ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course", "university.degree"]
default_options = ["no", "yes"]
housing_options = ["no", "yes"]
loan_options = ["no", "yes"]
contact_options = ["cellular", "telephone"]
month_options = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
day_options = ["mon", "tue", "wed", "thu", "fri"]
poutcome_options = ["failure", "nonexistent", "success"]

# Streamlit UI
st.set_page_config(page_title="Bank Subscription Predictor", layout="centered")
st.title("Bank Marketing Subscription Prediction")
st.markdown("""
This AI-powered tool predicts whether a client is likely to subscribe to a term deposit based on their profile and past campaign data.
""")

# Form inputs
with st.form("subscription_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 30)
        job = st.selectbox("Job", job_options)
        marital = st.selectbox("Marital Status", marital_options)
        education = st.selectbox("Education", education_options)
        default = st.selectbox("Credit in Default", default_options)
        housing = st.selectbox("Housing Loan", housing_options)
        loan = st.selectbox("Personal Loan", loan_options)
        contact = st.selectbox("Contact Type", contact_options)
    with col2:
        month = st.selectbox("Last Contact Month", month_options)
        day = st.selectbox("Last Contact Day", day_options)
        duration = st.number_input("Contact Duration (sec)", 0, 5000, 100)
        campaign = st.number_input("Campaign Contacts", 1, 100, 1)
        pdays = st.number_input("Days Since Last Contact", 0, 999, 999)
        previous = st.number_input("Previous Contacts", 0, 100, 0)
        poutcome = st.selectbox("Previous Outcome", poutcome_options)
        emp_var_rate = st.number_input("Employment Variation Rate", -5.0, 5.0, 1.1)
        cons_price_idx = st.number_input("Consumer Price Index", 90.0, 100.0, 93.2)
        cons_conf_idx = st.number_input("Consumer Confidence Index", -60.0, 0.0, -36.4)
        euribor3m = st.number_input("Euribor 3 Month Rate", 0.0, 10.0, 4.8)
        nr_employed = st.number_input("Number of Employees", 4000.0, 6000.0, 5191.0)

    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    input_dict = {
        'age': age,
        'job': job_options.index(job),
        'marital': marital_options.index(marital),
        'education': education_options.index(education),
        'default': default_options.index(default),
        'housing': housing_options.index(housing),
        'loan': loan_options.index(loan),
        'contact': contact_options.index(contact),
        'month': month_options.index(month),
        'day_of_week': day_options.index(day),
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome_options.index(poutcome),
        'emp.var.rate': emp_var_rate,
        'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m,
        'nr.employed': nr_employed,
        'campaign_per_previous': campaign / (previous + 1),
        'contact_month_combo': 0,  # unused in this version
        'loan_and_housing': 0     # unused in this version
    }

    input_df = pd.DataFrame([input_dict], columns=input_columns)
    input_df_scaled = scaler.transform(input_df)
    proba = model.predict_proba(input_df_scaled)[0][1]

    prediction = model.predict(input_df_scaled)[0]
    proba = model.predict_proba(input_df_scaled)[0][1]


    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"The client is likely to SUBSCRIBE. Probability: {proba:.2f}")
    else:
        st.error(f"The client is NOT likely to subscribe. Probability: {proba:.2f}")