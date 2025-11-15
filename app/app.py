import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------------------------
# Load Model + Features
# ------------------------------------------------------------------------------------------------
model = joblib.load("model/churn_model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

# ------------------------------------------------------------------------------------------------
# Custom CSS Styling (Elegant Dashboard)
# ------------------------------------------------------------------------------------------------
st.markdown("""
    <style>
        body {
            background-color: #f4f6f8;
        }
        .main {
            background-color: #f8fafc;
        }
        .title {
            font-size: 38px;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #566573;
            font-size: 16px;
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            border: none;
            font-size: 16px;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #155d8a;
        }
        .info-card, .danger-card {
            padding: 1rem;
            border-radius: 12px;
            text-align: center;
            font-size: 18px;
            font-weight: 600;
        }
        .info-card {
            background-color: #D5F5E3;
            color: #1D8348;
        }
        .danger-card {
            background-color: #FADBD8;
            color: #922B21;
        }
        .section-title {
            font-size: 22px;
            font-weight: 600;
            color: #2c3e50;
            margin-top: 20px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------------------------
st.sidebar.header("📘 About")
st.sidebar.write("This dashboard predicts telecom customer **churn** using ML + Explainable AI.")
st.sidebar.info("Dataset: IBM Telco Customer Churn")
st.sidebar.markdown("Built by *Customer Churn Team*")

# ------------------------------------------------------------------------------------------------
# Title Section
# ------------------------------------------------------------------------------------------------
st.markdown("<div class='title'>📊 Customer Churn Prediction Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Provide customer details below to get prediction.</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------------
# Input Form
# ------------------------------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input("🗓️ Tenure (months)", 0, 72, 12)
    internet_service = st.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    monthly = st.number_input("💰 Monthly Charges", 0.0, 150.0, 70.0)
    contract = st.selectbox("📄 Contract Type", ["Month-to-month", "One year", "Two year"])

with col3:
    payment_method = st.selectbox("💳 Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    paperless_billing = st.selectbox("📨 Paperless Billing", ["Yes", "No"])

# ------------------------------------------------------------------------------------------------
# Prepare Input Data
# ------------------------------------------------------------------------------------------------
raw = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "Contract": contract,
    "InternetService": internet_service,
    "PaymentMethod": payment_method,
    "PaperlessBilling": paperless_billing
}])

input_encoded = pd.get_dummies(raw)
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

# -------------------------------------------
# SHAP Setup (Robust)
# -------------------------------------------
try:
    inner_model = model.named_steps["model"]   # Pipeline case
except:
    inner_model = model                        # Non-pipeline model case

background = np.zeros((1, len(feature_columns)))

explainer = shap.Explainer(inner_model, background)


# ------------------------------------------------------------------------------------------------
# Predict Button
# ------------------------------------------------------------------------------------------------
if st.button("🔍 Predict Churn"):
    pred = model.predict(input_encoded)[0]
    prob = model.predict_proba(input_encoded)[0][1]
    conf = max(prob, 1 - prob)

    # -----------------------------------------------------------
    # Result Section
    # -----------------------------------------------------------
    st.markdown("---")
    res1, res2 = st.columns([1.2, 1])

    with res1:
        if pred == 1:
            st.markdown(f"<div class='danger-card'>⚠️ Customer is <b>Likely to Churn</b><br>Probability: {prob:.2%}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='info-card'>✅ Customer is <b>Likely to Stay</b><br>Retention: {(1 - prob):.2%}</div>", unsafe_allow_html=True)

    with res2:
        st.metric("Prediction Confidence", f"{conf*100:.1f}%")
        st.metric("Monthly Charges", f"${monthly:.2f}")
        st.metric("Tenure", f"{tenure} months")

    # -----------------------------------------------------------
    # Explanation Section
    # -----------------------------------------------------------
    st.markdown("<div class='section-title'>🧠 Why or Reason</div>", unsafe_allow_html=True)

    shap_vals = explainer(input_encoded)
    vals = shap_vals.values[0]

    contribs = pd.Series(vals, index=feature_columns)
    sorted_abs = contribs.abs().sort_values(ascending=False)
    top = sorted_abs.head(6).index

    # Capitalize "Tenure" + clean feature names
    def clean_feat(name):
        name = name.replace("_", " ").title()
        name = name.replace("Yes", "").strip()
        return name

    pos = contribs[top][contribs[top] > 0].sort_values(ascending=False)
    neg = contribs[top][contribs[top] < 0].sort_values()

    st.markdown("#### 🔺 Features Increasing Churn Risk")
    if len(pos) == 0:
        st.write("- None significant")
    else:
        for f, v in pos.items():
            st.write(f"- **{clean_feat(f)}** (impact: {v:.4f})")

    st.markdown("#### 🟢 Features Reducing Churn Risk")
    if len(neg) == 0:
        st.write("- None significant")
    else:
        for f, v in neg.items():
            st.write(f"- **{clean_feat(f)}** (impact: {v:.4f})")

    # -----------------------------------------------------------
    # SHAP Local Bar Plot
    # -----------------------------------------------------------
    st.markdown("<div class='section-title'>📊 Local Feature Impact</div>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    contrib_plot = contribs[top].sort_values()
    colors = ["#2ecc71" if v < 0 else "#e74c3c" for v in contrib_plot]
    sns.barplot(x=contrib_plot.values, y=[clean_feat(i) for i in contrib_plot.index], palette=colors, ax=ax)
    ax.set_title("Top Local Feature Impacts", fontsize=14)
    st.pyplot(fig)

# ------------------------------------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------------------------------------
st.markdown("---")
st.caption("Developed for Customer Churn Prediction • IBM Telco Dataset")
