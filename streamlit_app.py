import streamlit as st
import pandas as pd
import joblib

# Başlık
st.title(" Bank Term Deposit Prediction")
st.write("Bu uygulama, bir müşterinin bankanın vadeli mevduat teklifini kabul edip etmeyeceğini tahmin eder.")

# Modeli yükle
model = joblib.load("optimized_rf_model.pkl")

# Kullanıcıdan veri al
def user_input():
    age = st.slider("Age", 18, 95, 30)
    job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                               'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed'])
    marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
    education = st.selectbox("Education", ['basic.4y', 'basic.6y', 'basic.9y', 'high.school',
                                           'professional.course', 'university.degree', 'illiterate'])
    default = st.selectbox("Has Credit in Default?", ['no', 'yes'])
    housing = st.selectbox("Has Housing Loan?", ['no', 'yes'])
    loan = st.selectbox("Has Personal Loan?", ['no', 'yes'])
    contact = st.selectbox("Contact Type", ['cellular', 'telephone'])
    month = st.selectbox("Last Contact Month", ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    day_of_week = st.selectbox("Last Contact Day", ['mon', 'tue', 'wed', 'thu', 'fri'])

    duration = st.slider("Last Contact Duration (s)", 0, 5000, 300)
    campaign = st.slider("Number of Contacts", 1, 50, 1)
    pdays = st.slider("Days Since Last Contact", 0, 999, 999)
    previous = st.slider("Previous Contacts", 0, 10, 0)
    poutcome = st.selectbox("Previous Campaign Outcome", ['nonexistent', 'failure', 'success'])

    emp_var_rate = st.slider("Employment Variation Rate", -3.0, 2.0, 0.0)
    cons_price_idx = st.slider("Consumer Price Index", 92.0, 95.0, 93.0)
    cons_conf_idx = st.slider("Consumer Confidence Index", -50.0, -20.0, -40.0)
    euribor3m = st.slider("Euribor 3 Month Rate", 0.5, 5.0, 3.0)
    nr_employed = st.slider("Number of Employees", 4900.0, 5300.0, 5000.0)

    # Verileri DataFrame olarak döndür
    data = {
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'housing': housing, 'loan': loan, 'contact': contact,
        'month': month, 'day_of_week': day_of_week, 'duration': duration,
        'campaign': campaign, 'pdays': pdays, 'previous': previous,
        'poutcome': poutcome, 'emp.var.rate': emp_var_rate,
        'cons.price.idx': cons_price_idx, 'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m, 'nr.employed': nr_employed
    }
    return pd.DataFrame([data])

# Kullanıcı verisi al
input_df = user_input()

# Tahmin
if st.button("Tahmin Et"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success(" Bu müşteri Vade Mevduat teklifini KABUL EDEBİLİR.")
    else:
        st.error(" Bu müşteri Vade Mevduat teklifini büyük ihtimalle REDDEDECEK.")
