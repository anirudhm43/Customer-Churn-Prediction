# 📊 Customer Churn Prediction  

### 🧠 Project Overview  
This project aims to predict whether a telecom customer will discontinue the service (churn) based on their demographic details, service usage, and billing information.  
By identifying potential churners early, the company can take proactive measures to retain customers and improve business outcomes.  

---

## 🧾 Problem Statement  
Customer churn is when a customer stops using a company’s product or service.  
The goal of this project is to **build a machine learning model** that predicts the likelihood of customer churn and identifies key factors influencing it.

**Input:** Customer details like contract type, payment method, tenure, and charges.  
**Output:** Whether the customer is likely to churn (`Yes` / `No`).

---

## 📂 Dataset Information  
- **Name:** IBM Telco Customer Churn Dataset  
- **Source:** [Kaggle – Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)  
- **Format:** CSV file (7043 rows × 21 columns)  
- **Target Variable:** `Churn` (Yes = 1, No = 0)  
- **Description:** The dataset contains customer demographics, account information, service usage, and churn labels.
- **Detailed Report of the Dataset Analysis:** https://www.notion.so/Customer-Churn-Predictor-ML-Model-2972da7f7c2580ee8df7ce6b673587ca?source=copy_link

---

**Week 1:** :

## ⚙️ Frameworks & Tools Used  
| Category | Tools / Libraries |
|-----------|-------------------|
| **Language** | Python 3.10+ |
| **Environment** | Jupyter Notebook / VS Code |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Modeling** | Scikit-learn (Logistic Regression, Random Forest), XGBoost |
| **Explainability** | SHAP (SHapley Additive Explanations) |
| **UI / Deployment** | Streamlit |
| **Version Control** | Git, GitHub |

---

## 🧩 Project Components

**1️⃣ Dataset**  
- Contains customer demographics, billing, and churn information.  
- **Tools / Frameworks:** CSV file from Kaggle (IBM Telco Churn Dataset)

**2️⃣ Data Preprocessing**  
- Cleaning missing values, encoding categorical features, and normalizing numeric data.  
- **Tools / Frameworks:** Pandas, NumPy  

**3️⃣ Model Training & Evaluation**  
- Training ML models to predict churn and evaluating them using Accuracy, F1-Score, and ROC-AUC.  
- **Tools / Frameworks:** Scikit-learn, XGBoost  

**4️⃣ Explainability Analysis**  
- Identifying features that most influence churn predictions.  
- **Tools / Frameworks:** SHAP  

**5️⃣ User Interface / Deployment**  
- Simple web app for user input and instant churn prediction.  
- **Tools / Frameworks:** Streamlit  

**6️⃣ Version Control & Collaboration**  
- Managing versions, sharing code, and tracking progress.  
- **Tools / Frameworks:** Git, GitHub  

---

## 🧱 Project Architecture  
**Flow Overview:**  
Dataset (Telco Customer Churn)
↓
Data Preprocessing (Pandas, NumPy)
↓
Model Training & Evaluation (Scikit-learn, XGBoost)
↓
Explainability (SHAP)
↓
User Interface (Streamlit)



📁 **Architecture Diagram (Google Drive Link):**  
[👉 View Diagram](https://drive.google.com/file/d/1NHy53GrwCCu_6Q-NSx0H97zTjeFhq8oT/view?usp=drive_link)

---



**Week 2:**
 
customer_churn_project/
│
├── data/
│   ├── Telco-Customer-Churn.csv
│   ├── cleaned_churn_data.csv
│   └── feature_correlations.csv
│
├── notebooks/
│   ├── Week2_Day1_Data_Preprocessing_EDA.ipynb
│   ├── Week2_Day2_Baseline_Model.ipynb
│   └── Week2_Day3_Model_Comparison.ipynb
│
├── model/
│   └── churn_model.pkl
│
├── app/
│   └── app.py
│
├── results/
│   ├── eda_plots/
│   └── metrics_report.csv
│
├── README.md
├── requirements.txt
└── .gitignore

---


### Why This Structure?

**🧱 Separation of Concerns:** Keeps datasets, notebooks, models, and app components well organized for collaboration.

**🔁 Reproducibility:** requirements.txt and versioned notebooks ensure that experiments can be replicated easily.

**🛡️ Data Safety:** Raw data is never modified — cleaned versions are saved separately to preserve integrity.

**🚀 Deployability:** The app/ directory hosts a ready-to-deploy Streamlit web app powered by the saved model.

---

###  Data Preprocessing & EDA

### Data Preprocessing

The goal was to prepare raw data for reliable modeling by cleaning and transforming it into a consistent, machine-readable format.

**Key Steps:**
Handled missing or invalid values (notably in TotalCharges)
Encoded categorical variables and the target label (Churn → 1/0)
Verified correct data types and normalized key columns
Saved the processed dataset as cleaned_churn_data.csv

**✅ Outcome:**
A clean, structured, and bias-free dataset ready for training and feature analysis.

---


### Exploratory Data Analysis (EDA)

EDA helped uncover trends, patterns, and relationships in the data to guide model design.

**Highlights:**
Churn distribution: ~26% of customers churned (moderate imbalance)
Key features affecting churn: tenure, monthly charges, and contract type
Identified Top 10 correlated features with churn
Generated visual insights to support data-driven feature selection

**Purpose:**
Ensure the dataset is understood, trustworthy, and rich in actionable features before model development.

---

### Model Explainability – SHAP Analysis

After model training, SHAP (SHapley Additive exPlanations) was used to interpret feature importance and understand why customers churn.

**Top Influential Factors:**
- 📄 **Contract Type** – Customers with longer-term contracts (1–2 years) are far less likely to churn.
- ⏳ **Tenure** – Longer tenure strongly reduces churn; new users are more likely to leave.
- 💸 **Monthly Charges** – Higher monthly bills increase churn probability.
- 💳 **Payment Method** – Customers paying via electronic check churn more often.
- 🌐 **Internet Service Type** – Fiber optic users show higher churn, possibly due to higher costs.
- 🔒 **Online Security / Tech Support** – Availability of these add-ons helps retain customers.

These insights can help the business **design targeted retention strategies**, like:
- Offering discounts for high-value fiber customers.
- Encouraging longer contract sign-ups.
- Promoting bundled security or tech support services.


---

### Week 3 – Explainability + Streamlit App

### 🔍 Model Explainability (SHAP)

**SHAP was used to determine:**
Global Interpretability
**What features matter overall?**
Local Interpretability (Per Customer)
**Why did this specific person churn?**
Example local reasons:
High monthly charges
Short tenure
Month-to-month contract
These reasons are now displayed inside the Streamlit dashboard.

---

### 🖥️ Streamlit Web App (With Explainability)

**Features:**
Clean dashboard UI
Takes customer inputs
Predicts churn probability
Shows confidence
Displays Why or Reason
Shows SHAP local bar graph
Fully stylized UI

---

### 🚀 Deployment (Optional)

**Deployment options:**
Streamlit Cloud (free)
Render / HuggingFace Spaces
Docker-based deployment
GCP / AWS / Azure

(Current version deployed & tested on Streamlit Cloud.)

---


## 🚀 Planned Implementation Steps  
1. **Week 1:** Problem understanding, dataset and framework selection, architecture design.  
2. **Week 2:** Data preprocessing, model training, evaluation.  
3. **Week 3:** Explainability analysis, Streamlit app integration,final testing, report & deployment.

---

## 🔗 Review Details  
- **GitHub Repository:** https://github.com/anirudhm43/Customer-Churn-Prediction.git 
- **Dataset Link:** [Kaggle Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)  
- **Frameworks Identified:** Python, Pandas, Scikit-learn, XGBoost, SHAP, Streamlit  
- **Architecture Diagram (Drive):**https://drive.google.com/file/d/1684BqEauEvXbs5M2dRkIuRqaCSmsBkd5/view?usp=sharing
- **Project Folder Structure:**https://drive.google.com/file/d/1-a2dPpmQWah9Amqll2-oGb_CF039sO5H/view?usp=sharing
- **Detailed Report of the Dataset Analysis:** https://www.notion.so/Customer-Churn-Predictor-ML-Model-2972da7f7c2580ee8df7ce6b673587ca?source=copy_link


---

## 👥 Contributors  
- ANIRUDH M
- PRANAV K
- SHRIYA K
- SHRIYA MOHANTY 

---

### 🌟 Acknowledgment  
Dataset provided by **IBM** on **Kaggle** – used for educational and research purposes only.

---

