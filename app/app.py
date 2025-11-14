import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# ---------------------------------------------------------------
#  Page Configuration
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------
#  Load Model and Metadata
# ---------------------------------------------------------------
model = joblib.load("../model/churn_model.pkl")
feature_columns = joblib.load("../model/feature_columns.pkl")

# ---------------------------------------------------------------
#  Custom CSS Styling
# ---------------------------------------------------------------
st.markdown("""
    <style>
        .main {
            background-color: #f8fafc;
            padding: 2rem;
        }
        .stButton>button {
            background-color: #2E86C1;
            color: white;
            border-radius: 10px;
            font-size: 16px;
            padding: 0.5rem 1rem;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #1B4F72;
        }
        .result-success {
            background-color: #D5F5E3;
            padding: 1rem;
            border-radius: 10px;
            color: #145A32;
            text-align: center;
        }
        .result-danger {
            background-color: #FADBD8;
            padding: 1rem;
            border-radius: 10px;
            color: #78281F;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# 🧭 Sidebar
# ---------------------------------------------------------------
st.sidebar.header("📂 About the Project")
st.sidebar.write("This dashboard predicts whether a telecom customer is likely to **churn** based on demographics, billing, and service usage details.")
st.sidebar.markdown("---")
st.sidebar.info("👨‍💻 Built by Team Customer Churn Predictor")
st.sidebar.markdown("📊 Dataset: IBM Telco Customer Churn (Kaggle)")

# ---------------------------------------------------------------
#  App Title
# ---------------------------------------------------------------
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("#### Use the controls below to input customer details and predict churn likelihood.")

# ---------------------------------------------------------------
#  Input Section
# ---------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input("🗓️ Tenure (months)", min_value=0, max_value=72, value=12)
    internet_service = st.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])
with col2:
    monthly = st.number_input("💰 Monthly Charges", min_value=0.0, max_value=150.0, value=70.0)
    contract = st.selectbox("📄 Contract Type", ["Month-to-month", "One year", "Two year"])
with col3:
    payment_method = st.selectbox("💳 Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    paperless_billing = st.selectbox("📨 Paperless Billing", ["Yes", "No"])

# ---------------------------------------------------------------
#  Prediction Logic
# ---------------------------------------------------------------
input_data = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly],
    "Contract": [contract],
    "InternetService": [internet_service],
    "PaymentMethod": [payment_method],
    "PaperlessBilling": [paperless_billing]
})

# One-hot encode and align with training columns
input_encoded = pd.get_dummies(input_data)
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

# ---------------------------------------------------------------
#  Predict Button
# ---------------------------------------------------------------
if st.button("🔍 Predict Churn Probability"):
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    st.markdown("---")
    colA, colB = st.columns(2)

    # Show result box
    with colA:
        if prediction == 1:
            st.markdown(f"<div class='result-danger'><h3>⚠️ Customer is **Likely to Churn**</h3><p>Churn Probability: {probability:.2%}</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-success'><h3>✅ Customer is **Likely to Stay**</h3><p>Retention Probability: {(1 - probability):.2%}</p></div>", unsafe_allow_html=True)

    # Confidence and summary
    with colB:
        st.metric(label="Prediction Confidence", value=f"{max(probability, 1 - probability)*100:.1f} %")
        st.metric(label="Monthly Charges", value=f"${monthly:.2f}")
        st.metric(label="Tenure (months)", value=f"{tenure}")

    st.markdown("---")
    st.subheader(" Key Factors Considered")
    st.write("Tenure, Contract Type, Monthly Charges, Payment Method, Internet Service, and Paperless Billing influence the churn risk.")

# ---------------------------------------------------------------
# Footer
# ---------------------------------------------------------------
st.markdown("---")
st.caption("Developed as part of the Customer Churn Prediction Project • IBM Telco Dataset")
