# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="EMI Eligibility & Maximum Monthly EMI Prediction System", layout="centered")

# ---------------------------
# Feature metadata & options
# ---------------------------
FEATURE_ORDER = [
    'age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
    'family_size', 'dependents', 'school_fees', 'college_fees',
    'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
    'existing_loans', 'current_emi_amount', 'credit_score', 'bank_balance',
    'emergency_fund', 'requested_amount', 'requested_tenure',
    'emi_eligibility', 'max_monthly_emi', 'total_expenses',
    'debt_to_income', 'expense_to_income', 'estimated_affordability_ratio',
    'savings_ratio', 'combined_risk_score', 'income_credit_interaction',
    'employment_income_interaction', 'expense_loan_interaction',
    'gender_Female', 'gender_Male', 'marital_status_Married',
    'marital_status_Single', 'education_Graduate', 'education_High School',
    'education_Post Graduate', 'education_Professional',
    'employment_type_Government', 'employment_type_Private',
    'employment_type_Self-employed', 'company_type_Large Indian',
    'company_type_MNC', 'company_type_Mid-size', 'company_type_Small',
    'company_type_Startup', 'house_type_Family', 'house_type_Own',
    'house_type_Rented', 'emi_scenario_E-commerce Shopping EMI',
    'emi_scenario_Education EMI', 'emi_scenario_Home Appliances EMI',
    'emi_scenario_Personal Loan EMI', 'emi_scenario_Vehicle EMI',
    'credit_risk_level_High Risk', 'credit_risk_level_Low Risk',
    'credit_risk_level_Medium Risk', 'credit_risk_level_Very High Risk',
    'credit_risk_level_Very Low Risk', 'employment_stability_Moderate',
    'employment_stability_Stable', 'employment_stability_Unstable',
    'emi_eligibility_target'
]

# Categorical options (used for selectboxes)
CATEGORY_OPTIONS = {
    'gender': ['Female', 'Male'],
    'marital_status': ['Married', 'Single'],
    'education': ['Graduate', 'High School', 'Post Graduate', 'Professional'],
    'employment_type': ['Government', 'Private', 'Self-employed'],
    'company_type': ['Large Indian', 'MNC', 'Mid-size', 'Small', 'Startup'],
    'house_type': ['Family', 'Own', 'Rented'],
    'emi_scenario': ['E-commerce Shopping EMI', 'Education EMI', 'Home Appliances EMI', 'Personal Loan EMI',
                     'Vehicle EMI'],
    'credit_risk_level': ['High Risk', 'Low Risk', 'Medium Risk', 'Very High Risk', 'Very Low Risk'],
    'employment_stability': ['Moderate', 'Stable', 'Unstable'],
    'Existing_loan': ['Yes', 'No']
}

NUMERIC_COLS_TO_SCALE = [
    'monthly_salary', 'monthly_rent', 'school_fees', 'college_fees',
    'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
    'existing_loans', 'current_emi_amount', 'bank_balance',
    'emergency_fund', 'requested_amount', 'requested_tenure',
    'total_expenses', 'debt_to_income', 'expense_to_income',
    'estimated_affordability_ratio', 'savings_ratio',
    'combined_risk_score', 'income_credit_interaction',
    'employment_income_interaction', 'expense_loan_interaction'
]


# ---------------------------
# Helpers
# ---------------------------
def load_joblib_from_upload(uploaded_file):
    try:
        if uploaded_file is None:
            return None
        return joblib.load(uploaded_file)
    except Exception as e:
        st.warning(f"Failed to load joblib: {e}")
        return None


def apply_feature_engineering(fe):
    """
    Apply exact feature-engineering code you provided to dataframe 'fe' (single-row).
    """
    eps = 1e-6

    # total_expenses
    fe['total_expenses'] = (
            fe['monthly_rent'].fillna(0) +
            fe['school_fees'].fillna(0) +
            fe['college_fees'].fillna(0) +
            fe['travel_expenses'].fillna(0) +
            fe['groceries_utilities'].fillna(0) +
            fe['other_monthly_expenses'].fillna(0) +
            fe['current_emi_amount'].fillna(0)
    )


    # existing_loans
    fe['existing_loans']=fe['existing_loans'].map({
       'Yes':1.0,
        'No':0.0
    })
    # debt_to_income
    fe['debt_to_income'] = fe['existing_loans'] / (fe['monthly_salary'] + eps)

    # expense_to_income
    fe['expense_to_income'] = fe['total_expenses'] / (fe['monthly_salary'] + eps)

    # estimated_affordability_ratio
    fe['estimated_affordability_ratio'] = (
            (fe['monthly_salary'] - fe['other_monthly_expenses']) /
            (fe['requested_amount'] / fe['requested_tenure'] + eps)
    )

    # savings_ratio
    fe['savings_ratio'] = (fe['bank_balance'] + fe['emergency_fund']) / (fe['monthly_salary'] + eps)

    # credit_risk_level
    def credit_risk(score):
        if pd.isna(score):
            return 'Unknown'
        elif score >= 800:
            return 'Very Low Risk'
        elif score >= 700:
            return 'Low Risk'
        elif score >= 600:
            return 'Medium Risk'
        elif score >= 500:
            return 'High Risk'
        else:
            return 'Very High Risk'

    fe['credit_risk_level'] = fe['credit_score'].apply(credit_risk)

    # employment_stability
    def employment_stability(years):
        if years >= 5:
            return 'Stable'
        elif years >= 2:
            return 'Moderate'
        else:
            return 'Unstable'

    fe['employment_stability'] = fe['years_of_employment'].apply(employment_stability)

    # combined_risk_score
    risk_map = {
        'Very Low Risk': 1,
        'Low Risk': 2,
        'Medium Risk': 3,
        'High Risk': 4,
        'Very High Risk': 5,
        'Unknown': np.nan
    }
    stability_map = {'Stable': 1, 'Moderate': 2, 'Unstable': 3}

    fe['combined_risk_score'] = (
            fe['credit_risk_level'].map(risk_map) +
            fe['employment_stability'].map(stability_map)
    )

    # interaction features
    fe['income_credit_interaction'] = fe['monthly_salary'] * fe['credit_score']
    fe['employment_income_interaction'] = fe['years_of_employment'] * fe['monthly_salary']
    fe['expense_loan_interaction'] = fe['total_expenses'] * fe['existing_loans']

    return fe


def one_hot_encode_options(fe):
    """
    Create one-hot columns for our categorical options. For each category,
    create all possible one-hot columns listed in FEATURE_ORDER and set 1 for chosen option.
    """
    # gender
    for g in CATEGORY_OPTIONS['gender']:
        col = f"gender_{g}"
        fe[col] = (fe['gender'] == g).astype(float)
    fe.drop(columns=['gender'], inplace=True)

    # marital_status
    for m in CATEGORY_OPTIONS['marital_status']:
        col = f"marital_status_{m}"
        fe[col] = (fe['marital_status'] == m).astype(float)
    fe.drop(columns='marital_status', axis=1, inplace=True)

    # education
    for e in CATEGORY_OPTIONS['education']:
        col = f"education_{e}"
        fe[col] = (fe['education'] == e).astype(float)
    fe.drop(columns='education', axis=1, inplace=True)

    # employment_type
    for et in CATEGORY_OPTIONS['employment_type']:
        col = f"employment_type_{et}"
        fe[col] = (fe['employment_type'] == et).astype(float)
    fe.drop(columns='employment_type', axis=1, inplace=True)

    # company_type
    for c in CATEGORY_OPTIONS['company_type']:
        col = f"company_type_{c}"
        fe[col] = (fe['company_type'] == c).astype(float)
    fe.drop(columns='company_type', axis=1, inplace=True)

    # house_type
    for h in CATEGORY_OPTIONS['house_type']:
        col = f"house_type_{h}"
        fe[col] = (fe['house_type'] == h).astype(float)
    fe.drop(columns='house_type', axis=1, inplace=True)

    # emi_scenario
    for s in CATEGORY_OPTIONS['emi_scenario']:
        col = f"emi_scenario_{s}"
        fe[col] = (fe['emi_scenario'] == s).astype(float)
    fe.drop(columns='emi_scenario', axis=1, inplace=True)

    # credit_risk_level - map to options (note: our credit_risk returns 'Very Low Risk' etc)
    for cr in CATEGORY_OPTIONS['credit_risk_level']:
        col = f"credit_risk_level_{cr}"
        fe[col] = (fe['credit_risk_level'] == cr).astype(float)
    fe.drop(columns='credit_risk_level', axis=1, inplace=True)

    # employment_stability
    for stbl in CATEGORY_OPTIONS['employment_stability']:
        col = f"employment_stability_{stbl}"
        fe[col] = (fe['employment_stability'] == stbl).astype(float)
    fe.drop(columns='employment_stability', axis=1, inplace=True)


    return fe


def build_final_feature_vector(fe):
    """
    Ensure all columns in FEATURE_ORDER exist, fill missing with 0, and return in order.
    """
    # create missing columns with 0
    for c in FEATURE_ORDER:
        if c not in fe.columns:
            fe[c] = 0.0

    # Keep only FEATURE_ORDER in that order
    X = fe[FEATURE_ORDER].fillna(0).astype(float).values.reshape(1, -1)
    return X


# ---------------------------
# UI form
# ---------------------------
st.title("EMI Eligibility & Maximum Monthly EMI Prediction System")
st.write(
    "Fill applicant information's...")

with st.form("input_form"):
    st.header("Applicant Info")
    col1, col2 = st.columns(2)

    with col2:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        monthly_salary = st.number_input("Monthly Salary (₹)", min_value=0.0, value=40000.0, step=1000.0)
        years_of_employment = st.number_input("Years of Employment", min_value=0.0, value=3.0)
        monthly_rent = st.number_input("Monthly Rent (₹)", min_value=0.0, value=10000.0)
        family_size = st.number_input("Family Size", min_value=1, value=3)
        dependents = st.number_input("Dependents", min_value=0, value=1)
        school_fees = st.number_input("Monthly School Fees (₹)", min_value=0.0, value=0.0)
        college_fees = st.number_input("Monthly College Fees (₹)", min_value=0.0, value=0.0)

    with col1:
        travel_expenses = st.number_input("Monthly Travel Expenses (₹)", min_value=0.0, value=2000.0)
        groceries_utilities = st.number_input("Groceries & Utilities (₹)", min_value=0.0, value=8000.0)
        other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)", min_value=0.0, value=2000.0)
        existing_loans = st.selectbox("Existing_loan", options=CATEGORY_OPTIONS['Existing_loan'])
        current_emi_amount = st.number_input("Current EMI Amount (₹)", min_value=0.0, value=2500.0)
        credit_score = st.number_input("Credit Score (300-900)", min_value=300, max_value=900, value=650)
        bank_balance = st.number_input("Bank Balance (₹)", min_value=0.0, value=20000.0)
        emergency_fund = st.number_input("Emergency Fund (₹)", min_value=0.0, value=20000.0)

    requested_amount = st.number_input("Requested Loan Amount (₹)", min_value=0.0, value=200000.0, step=5000.0)
    requested_tenure = st.number_input("Requested Tenure (months)", min_value=1, max_value=360, value=36)

    st.markdown("---")
    st.subheader("Personal Background")
    gender = st.selectbox("Gender", options=CATEGORY_OPTIONS['gender'])
    marital_status = st.selectbox("Marital Status", options=CATEGORY_OPTIONS['marital_status'])
    education = st.selectbox("Highest Education", options=CATEGORY_OPTIONS['education'])
    employment_type = st.selectbox("Employment Type", options=CATEGORY_OPTIONS['employment_type'])
    company_type = st.selectbox("Company Type", options=CATEGORY_OPTIONS['company_type'])
    house_type = st.selectbox("House Type", options=CATEGORY_OPTIONS['house_type'])
    emi_scenario = st.selectbox("EMI Scenario", options=CATEGORY_OPTIONS['emi_scenario'])

    submit = st.form_submit_button("Compute")

# ---------------------------
# File uploads: optional scaler & models
# ---------------------------

#StandardScaler sclaer from sklearn.preprocessing
#Emi_model (XGBOOST Tuned model with accuracy:-(~ 97 %))
#max_emi_model (XGBoost untuned)

scaler_path="scaler.joblib"
emi_model_path="XGBoost_tuned.joblib"
max_emi_model_path="XGBoost.joblib"

scaler = load_joblib_from_upload(scaler_path)
emi_model = load_joblib_from_upload(emi_model_path)
max_emi_model = load_joblib_from_upload(max_emi_model_path)

if submit:
    # build single-row df 'fe' from inputs
    data = {
        'age': age,
        'monthly_salary': monthly_salary,
        'years_of_employment': years_of_employment,
        'monthly_rent': monthly_rent,
        'family_size': family_size,
        'dependents': dependents,
        'school_fees': school_fees,
        'college_fees': college_fees,
        'travel_expenses': travel_expenses,
        'groceries_utilities': groceries_utilities,
        'other_monthly_expenses': other_monthly_expenses,
        'existing_loans': existing_loans,
        'current_emi_amount': current_emi_amount,
        'credit_score': credit_score,
        'bank_balance': bank_balance,
        'emergency_fund': emergency_fund,
        'requested_amount': requested_amount,
        'requested_tenure': requested_tenure,
        # categorical placeholders (used in one_hot)
        'gender': gender,
        'marital_status': marital_status,
        'education': education,
        'employment_type': employment_type,
        'company_type': company_type,
        'house_type': house_type,
        'emi_scenario': emi_scenario
    }

    fe = pd.DataFrame([data])

    # apply exactly your feature engineering (same code you provided)
    fe = apply_feature_engineering(fe)

    # after computing credit_risk_level and employment_stability, apply one-hot encoding
    fe = one_hot_encode_options(fe)

    # Optionally scale numeric columns if scaler is provided
    if scaler is not None:
        # Ensure scaler supports transform on DataFrame subset - typical sklearn scaler does.
        try:
            # scaler expects 2D array with columns in specific order used during training.
            # We try to transform NUMERIC_COLS_TO_SCALE intersection with df
            cols_to_scale = [c for c in NUMERIC_COLS_TO_SCALE if c in fe.columns]
            fe_vals = fe[cols_to_scale].fillna(0).astype(float)
            scaled_vals = scaler.transform(fe_vals)
            fe.loc[:, cols_to_scale] = scaled_vals
            # st.success("Scaler applied to numeric columns.")
        except Exception as e:
            st.warning(f"Could not apply scaler: {e}. Proceeding without scaling.")
    else:
        st.info("No scaler uploaded — raw numeric features will be used.")



    # Show processed features

    # Predict eligibility (classification)
    eligible = None
    if emi_model is not None:
        try:
            if hasattr(emi_model, "predict_proba"):
                prob = emi_model.predict_proba(fe)[0][1]
                eligible = bool(emi_model.predict_proba(fe)[0][1] >= 0.4 or emi_model.predict_proba(fe)[0][2] >= 0.4)

                st.caption("🔍 Prediction Summary")

                st.write(f"* Not Eligible - probability:{emi_model.predict_proba(fe)[0][0]:.2f}")
                st.write(f"* Eligible - probability:{emi_model.predict_proba(fe)[0][1]:.2f}")
                st.write(f"* High_Risk - probability:{emi_model.predict_proba(fe)[0][2]:.2f}")

            else:
                pred = emi_model.predict(fe)
                eligible = bool(pred[0])
        except Exception as e:
            st.error(f"Error running eligibility model: {e}")
            eligible = None
    else:
        st.info(
            "No EMI eligibility model uploaded. Upload emi_eligibility_model.joblib in the sidebar to enable model prediction.")

    # Predict max monthly EMI (regression)
    max_emi = None
    if max_emi_model is not None:
        try:
            pred_val = max_emi_model.predict(fe)
            max_emi = float(pred_val[0])
        except Exception as e:
            st.error(f"Error running max EMI model: {e}")
            max_emi = None
    else:
        st.info("No max EMI model uploaded. Upload max_emi_model.joblib in the sidebar to enable regression.")

    # Show results
    st.markdown("---")
    st.subheader("Result")
    if eligible is None:
        st.warning("Eligibility not determined (model missing or failed).")
    else:
        if eligible:
            st.success("✅ Applicant is ELIGIBLE for EMI.")
            if max_emi is not None:
                st.metric("Estimated Maximum Monthly EMI (₹)", f"{max_emi:,.2f}")
            else:
                st.info("Max EMI not provided (regression model missing).")
        else:
            st.error("❌ Applicant is NOT eligible for EMI based on prediction.")
            st.write("Suggestions to improve eligibility:")
            st.write("- Reduce existing debts / current EMIs")
            st.write("- Increase tenure / reduce requested amount")
            st.write("- Improve credit score and emergency savings")

footer = """
   <style>
   .footer {
       position: fixed;
       left: 0;
       bottom: 0;
       width: 100%;
       background: linear-gradient(90deg, #0A0A0A, #1A1A1A);
       padding: 12px;
       color: #FFFFFF;
       text-align: center;
       font-size: 14px;
       font-family: 'Segoe UI', sans-serif;
       border-top: 1px solid #333;
       box-shadow: 0px -2px 10px rgba(0,0,0,0.3);
       z-index: 999;
   }

   .footer a {
       color: #4CAEF2;
       text-decoration: none;
       font-weight: 500;
   }

   .footer a:hover {
       color: #77C6FF;
       text-decoration: underline;
   }
   </style>

   <div class="footer">
       <p>🔍 AI-Powered EMI Prediction App • Built for Smarter Financial Decisions |
       <a href="https://github.com/" target="_blank">GitHub</a></p>
   </div>
   """

st.markdown(footer, unsafe_allow_html=True)
