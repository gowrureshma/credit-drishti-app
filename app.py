import streamlit as st
import requests
import json
import pandas as pd
import numpy as np

# --- v2 App Configuration ---
st.set_page_config(page_title="Credit-Drishti v2.0", layout="wide", initial_sidebar_state="expanded")
st.title("Credit-Drishti v2.0: Scoring Engine")

# --- API Endpoint ---
API_URL = "https://credit-drishti-app.onrender.com/predict"

# --- User Input Form in Sidebar ---
with st.sidebar:
    st.header("Applicant Details")
    with st.form("loan_application_form"):
        with st.expander("Personal Information", expanded=True):
            person_age = st.number_input("Age", 18, 100, 30)
            person_income = st.number_input("Annual Income (INR)", 4000, 20000000, 200000)
            person_emp_length = st.number_input("Employment Length (Years)", 0, 50, 5)
            person_home_ownership = st.selectbox("Home Ownership", ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
        
        with st.expander("Loan Details", expanded=True):
            loan_intent = st.selectbox("Loan Purpose", ['EDUCATION', 'PERSONAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION', 'MEDICAL'])
            loan_grade = st.selectbox("Loan Grade", ['B', 'A', 'C', 'D', 'E', 'F', 'G'])
            loan_amnt = st.number_input("Loan Amount (INR)", 500, 10000000, 50000)
            loan_int_rate = st.number_input("Interest Rate (%)", 5.0, 25.0, 10.0, step=0.1)
        
        with st.expander("Credit History", expanded=True):
            cb_person_default_on_file = st.selectbox("Has Defaulted Before?", ['N', 'Y'])
            cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", 0, 40, 7)

        loan_percent_income = round(loan_amnt / person_income, 2) if person_income > 0 else 0
        st.info(f"Loan to Income Ratio: {loan_percent_income:.2f}")
        submitted = st.form_submit_button("Assess Creditworthiness", use_container_width=True)

# --- API Call and Result Display ---
if submitted:
    payload = {
        'person_age': person_age, 'person_income': person_income, 'person_home_ownership': person_home_ownership,
        'person_emp_length': person_emp_length, 'loan_intent': loan_intent, 'loan_grade': loan_grade,
        'loan_amnt': loan_amnt, 'loan_int_rate': float(loan_int_rate), 'loan_percent_income': loan_percent_income,
        'cb_person_default_on_file': cb_person_default_on_file, 'cb_person_cred_hist_length': cb_person_cred_hist_length
    }
    
    try:
        with st.spinner('Assessing applicant...'):
            response = requests.post(API_URL, data=json.dumps(payload), headers={'Content-Type': 'application/json'}, timeout=10)
        
        if response.status_code == 200:
            results = response.json()
            st.success("Assessment Complete!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Final Decision", results['decision'])
            col2.metric("Risk Probability", f"{results['risk_probability_percent']}%")
            col3.metric("Scorecard Score", results['scorecard_score'])

            st.markdown("---")
            st.subheader("Interpretable Scorecard Breakdown")
            
            score_data = {k: v for k, v in results['scorecard_breakdown'].items() if k != 'Base Score'}
            base_score = results['scorecard_breakdown'].get('Base Score', 0)
            
            df_score = pd.DataFrame(list(score_data.items()), columns=['Feature Contribution', 'Points'])
            
            df_score['Positive'] = np.where(df_score['Points'] >= 0, df_score['Points'], 0)
            df_score['Negative'] = np.where(df_score['Points'] < 0, df_score['Points'], 0)
            df_score.set_index('Feature Contribution', inplace=True)

            st.write(f"The final score of **{results['scorecard_score']}** is calculated from a **Base Score** of **{base_score}** plus the following adjustments:")
            st.bar_chart(df_score[['Positive', 'Negative']], color=["#2ca02c", "#d62728"])
            
            st.info("The **Final Decision** and **Risk Probability** are determined by a high-accuracy LightGBM model, while the **Scorecard** provides full explainability.", icon="💡")
        else:
            st.error(f"API Error (Status {response.status_code}): {response.text}")
            
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: Could not connect to the API server at {API_URL}. Please ensure the server is running and accessible.")
else:
    st.info("Please fill in the applicant details on the left and click 'Assess Creditworthiness'.")
