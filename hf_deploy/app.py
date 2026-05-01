# hf_deploy/app.py
"""
Streamlit Interactive Dashboard for Customer Churn Prediction.

Run with:
    streamlit run hf_deploy/app.py

The dashboard allows a user to enter customer details interactively
and get an instant churn prediction with SHAP explanation.
"""

import sys
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for model_path in [
        os.path.join(base_dir, 'models', 'best_model.pkl'),
        os.path.join(base_dir, '..', 'models', 'best_model.pkl'),
    ]:
        if os.path.exists(model_path):
            return joblib.load(model_path)
    raise FileNotFoundError("Model not found. Check models/ folder.")


pipeline = load_model()
preprocessor = pipeline.named_steps['preprocessor']
classifier = pipeline.named_steps['classifier']


# ── Feature engineering ───────────────────────────────────────────────────────
def apply_feature_engineering(df):
    df = df.copy()

    # charges_per_month
    def calc_charges(row):
        if row['tenure_months'] == 0:
            return row['monthly_charges']
        return round(row['total_charges'] / row['tenure_months'], 2)
    df['charges_per_month'] = df.apply(calc_charges, axis=1)

    # tenure_group_encoded
    def get_tenure_group(tenure):
        if tenure <= 12:
            return 0
        elif tenure <= 24:
            return 1
        elif tenure <= 48:
            return 2
        else:
            return 3
    df['tenure_group_encoded'] = df['tenure_months'].apply(get_tenure_group)

    # encode binary columns
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    for col in ['gender', 'partner', 'dependents', 'phone_service', 'paperless_billing']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map(binary_map)

    return df


def get_feature_names():
    from src.features import NUMERIC_COLS, MULTI_CAT_COLS, BINARY_COLS
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_names = ohe.get_feature_names_out(MULTI_CAT_COLS).tolist()
    binary_present = [c for c in BINARY_COLS if c in [
        'gender', 'senior_citizen', 'partner', 'dependents',
        'phone_service', 'paperless_billing'
    ]]
    return NUMERIC_COLS + cat_names + binary_present


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Customer Details")
st.sidebar.markdown("Fill in the customer information below.")

st.sidebar.header("Account Information")
tenure = st.sidebar.slider("Tenure (months)", min_value=0, max_value=72, value=12)
contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)
payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

st.sidebar.header("Charges")
monthly_charges = st.sidebar.slider(
    "Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0, step=0.5
)
default_total = float(monthly_charges * tenure) if tenure > 0 else float(monthly_charges)
total_charges = st.sidebar.number_input(
    "Total Charges ($)", min_value=0.0, max_value=10000.0,
    value=default_total, step=10.0
)

st.sidebar.header("Internet Services")
internet_service = st.sidebar.selectbox(
    "Internet Service", ["Fiber optic", "DSL", "No"]
)
online_security = st.sidebar.selectbox(
    "Online Security", ["No", "Yes", "No internet service"]
)
online_backup = st.sidebar.selectbox(
    "Online Backup", ["No", "Yes", "No internet service"]
)
device_protection = st.sidebar.selectbox(
    "Device Protection", ["No", "Yes", "No internet service"]
)
tech_support = st.sidebar.selectbox(
    "Tech Support", ["No", "Yes", "No internet service"]
)
streaming_tv = st.sidebar.selectbox(
    "Streaming TV", ["No", "Yes", "No internet service"]
)
streaming_movies = st.sidebar.selectbox(
    "Streaming Movies", ["No", "Yes", "No internet service"]
)

st.sidebar.header("Phone Services")
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox(
    "Multiple Lines", ["No", "Yes", "No phone service"]
)

st.sidebar.header("Demographics")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen_input = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])


# ── Build input DataFrame ─────────────────────────────────────────────────────
senior_citizen_val = 1 if senior_citizen_input == "Yes" else 0

input_data = pd.DataFrame([{
    'gender': gender,
    'senior_citizen': senior_citizen_val,
    'partner': partner,
    'dependents': dependents,
    'tenure_months': tenure,
    'phone_service': phone_service,
    'multiple_lines': multiple_lines,
    'internet_service': internet_service,
    'online_security': online_security,
    'online_backup': online_backup,
    'device_protection': device_protection,
    'tech_support': tech_support,
    'streaming_tv': streaming_tv,
    'streaming_movies': streaming_movies,
    'contract': contract,
    'paperless_billing': paperless_billing,
    'payment_method': payment_method,
    'monthly_charges': monthly_charges,
    'total_charges': total_charges,
}])

input_processed = apply_feature_engineering(input_data)

# ── Run prediction ────────────────────────────────────────────────────────────
churn_prob = pipeline.predict_proba(input_processed)[0][1]
churn_pred = "Yes" if churn_prob >= 0.5 else "No"

if churn_prob >= 0.7:
    risk_level = "HIGH RISK"
    risk_color = "#FF4B4B"
    action = "Immediate retention action recommended. Consider offering a contract upgrade or discount."
elif churn_prob >= 0.4:
    risk_level = "MEDIUM RISK"
    risk_color = "#FFA500"
    action = "Proactive outreach recommended. Monitor this customer closely."
else:
    risk_level = "LOW RISK"
    risk_color = "#00C853"
    action = "Customer appears stable. No immediate action required."


# ── Main panel ────────────────────────────────────────────────────────────────
st.title("Customer Churn Prediction Dashboard")
st.markdown("*Powered by XGBoost · AUC-ROC 0.8535 · IBM Telco Dataset*")
st.markdown("---")

# Top metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Churn Prediction",
        value=churn_pred,
        delta="Will churn" if churn_pred == "Yes" else "Will stay"
    )

with col2:
    avg_churn = 0.265
    delta_val = churn_prob - avg_churn
    st.metric(
        label="Churn Probability",
        value=f"{churn_prob:.1%}",
        delta=f"{delta_val:+.1%} vs avg"
    )

with col3:
    st.metric(label="Risk Level", value=risk_level)

with col4:
    if tenure <= 12:
        tenure_group = "New (0-12 mo)"
    elif tenure <= 24:
        tenure_group = "Developing (13-24 mo)"
    elif tenure <= 48:
        tenure_group = "Established (25-48 mo)"
    else:
        tenure_group = "Loyal (49+ mo)"
    st.metric(label="Tenure Group", value=tenure_group)

# Risk banner
st.markdown(
    "<div style='background-color:{color}22; border-left: 5px solid {color}; "
    "padding: 15px; border-radius: 5px; margin: 10px 0;'>"
    "<b style='color:{color}; font-size:18px;'>{level}</b><br>{action}"
    "</div>".format(color=risk_color, level=risk_level, action=action),
    unsafe_allow_html=True
)

st.markdown("---")

# Two columns: gauge + summary table
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Churn Probability Gauge")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(["Churn Risk"], [1.0], color="#e0e0e0", height=0.5)

    if churn_prob >= 0.7:
        bar_color = "#FF4B4B"
    elif churn_prob >= 0.4:
        bar_color = "#FFA500"
    else:
        bar_color = "#00C853"

    ax.barh(["Churn Risk"], [churn_prob], color=bar_color, height=0.5)
    ax.axvline(x=0.5, color="black", linestyle="--", linewidth=1.5, label="Threshold (50%)")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Churn Probability")
    ax.text(min(churn_prob + 0.02, 0.92), 0, f"{churn_prob:.1%}",
            va="center", fontweight="bold", fontsize=12)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Predicted Churn Probability")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_right:
    st.subheader("Customer Profile Summary")

    if contract == "Month-to-month":
        contract_risk = "High risk"
    else:
        contract_risk = "Low risk"

    if tenure <= 12:
        tenure_risk = "New customer"
    else:
        tenure_risk = "Established"

    if monthly_charges > 65:
        charges_risk = "Above average"
    else:
        charges_risk = "Below average"

    if internet_service == "Fiber optic":
        internet_risk = "Higher churn service"
    else:
        internet_risk = "Lower risk"

    if online_security == "No":
        security_risk = "No protection"
    else:
        security_risk = "Protected"

    if tech_support == "No":
        support_risk = "No support"
    else:
        support_risk = "Supported"

    if payment_method == "Electronic check":
        payment_risk = "Higher risk method"
    else:
        payment_risk = "Lower risk method"

    if dependents == "Yes":
        dependents_risk = "Lower risk"
    else:
        dependents_risk = "Higher risk"

    summary_data = {
        "Feature": [
            "Contract", "Tenure", "Monthly Charges",
            "Internet Service", "Online Security",
            "Tech Support", "Payment Method", "Dependents"
        ],
        "Value": [
            contract,
            f"{tenure} months",
            f"${monthly_charges:.2f}",
            internet_service,
            online_security,
            tech_support,
            payment_method,
            dependents
        ],
        "Risk Assessment": [
            contract_risk,
            tenure_risk,
            charges_risk,
            internet_risk,
            security_risk,
            support_risk,
            payment_risk,
            dependents_risk
        ]
    }

    st.dataframe(
        pd.DataFrame(summary_data),
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")

# ── SHAP Explanation ──────────────────────────────────────────────────────────
st.subheader("Why did the model make this prediction? (SHAP Explanation)")
st.markdown(
    "Red bars pushed the churn probability **up**. "
    "Blue bars pushed it **down**. "
    "The starting point is the average prediction across all customers (26.5%)."
)

try:
    feature_names = get_feature_names()
    X_transformed = preprocessor.transform(input_processed)
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed_df)

    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_transformed_df.iloc[0],
        feature_names=feature_names
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(explanation, max_display=12, show=False)
    plt.title("Feature Contributions to This Prediction", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

except Exception as e:
    st.warning(f"SHAP explanation could not be generated: {e}")
    st.info("The prediction above is still valid. Only the explanation chart is affected.")

st.markdown("---")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center; color:grey; font-size:12px;'>"
    "Customer Churn Prediction System · XGBoost + SHAP + Streamlit · "
    "IBM Telco Dataset · AUC-ROC: 0.8535"
    "</div>",
    unsafe_allow_html=True
)