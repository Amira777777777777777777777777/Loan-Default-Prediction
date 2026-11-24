
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# -----------------------------------
# 1. Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

# -----------------------------------
# 2. Load Model and Settings
# -----------------------------------

@st.cache_resource
def load_model():
    return joblib.load(
        BASE_DIR / "loan_default_model.pkl"
    )

model = load_model()

with open(BASE_DIR / "model_info.json") as file:
    model_info = json.load(file)

with open(BASE_DIR / "app_settings.json") as file:
    app_settings = json.load(file)

categories = app_settings["categorical_options"]
numbers = app_settings["numerical_settings"]

# -----------------------------------
# 3. Application Design
# -----------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #F4F7FB;
    color: #14243B;
}

h1, h2, h3 {
    color: #14243B !important;
}

[data-testid="stSidebar"] {
    background-color: #E4EBF5;
}

.stButton > button {
    background-color: #14243B;
    color: white;
    border-radius: 10px;
    width: 100%;
    padding: 12px;
}

.stButton > button:hover {
    background-color: #294B75;
    color: white;
}

.header-box {
    background-color: #14243B;
    padding: 30px;
    border-radius: 15px;
    color: white;
    margin-bottom: 25px;
}

.header-box h1 {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# 4. Application Header
# -----------------------------------

st.markdown("""
<div class="header-box">
    <h1>Loan Default Prediction</h1>
    <p>Machine Learning-Based Credit Risk Assessment</p>
    <p>Powered by Random Forest</p>
</div>
""", unsafe_allow_html=True)

st.write(
    "Enter borrower information below "
    "to generate a loan default prediction."
)

# -----------------------------------
# 5. Sidebar
# -----------------------------------

st.sidebar.title("About the Project")

st.sidebar.write(
    "Data Science for Business Applications"
)

st.sidebar.write(
    "Higher Colleges of Technology"
)

st.sidebar.divider()

st.sidebar.subheader("Model Information")

st.sidebar.write(
    "Selected Model:",
    model_info["model_name"]
)

st.sidebar.metric(
    "Test Accuracy",
    f'{model_info["test_accuracy"]:.2%}'
)

st.sidebar.metric(
    "F1-score",
    f'{model_info["test_f1_score"]:.4f}'
)

st.sidebar.metric(
    "ROC-AUC",
    f'{model_info["test_roc_auc"]:.4f}'
)

st.sidebar.divider()

st.sidebar.caption(
    "Educational demonstration only."
)

# -----------------------------------
# 6. Input Functions
# -----------------------------------

def numeric_input(column, label, integer=False):

    settings = numbers[column]

    if integer:
        return st.number_input(
            label,
            min_value=int(settings["min"]),
            max_value=int(settings["max"]),
            value=int(settings["default"]),
            step=1
        )

    return st.number_input(
        label,
        min_value=float(settings["min"]),
        max_value=float(settings["max"]),
        value=float(settings["default"]),
        format="%.2f"
    )


def category_input(column, label):

    return st.selectbox(
        label,
        categories[column]
    )

# -----------------------------------
# 7. Personal Information
# -----------------------------------

st.subheader("1. Personal Information")

col1, col2, col3 = st.columns(3)

with col1:

    age = numeric_input(
        "Age", "Age", integer=True
    )

    education = category_input(
        "Education", "Education Level"
    )

with col2:

    employment = category_input(
        "EmploymentType", "Employment Type"
    )

    marital_status = category_input(
        "MaritalStatus", "Marital Status"
    )

with col3:

    dependents = category_input(
        "HasDependents", "Has Dependents?"
    )

    months_employed = numeric_input(
        "MonthsEmployed",
        "Months Employed",
        integer=True
    )

st.divider()

# -----------------------------------
# 8. Financial Information
# -----------------------------------

st.subheader("2. Financial Information")

col1, col2, col3 = st.columns(3)

with col1:

    income = numeric_input(
        "Income", "Annual Income"
    )

    credit_score = numeric_input(
        "CreditScore",
        "Credit Score",
        integer=True
    )

with col2:

    credit_lines = numeric_input(
        "NumCreditLines",
        "Number of Credit Lines",
        integer=True
    )

    dti_ratio = numeric_input(
        "DTIRatio",
        "Debt-to-Income Ratio"
    )

with col3:

    mortgage = category_input(
        "HasMortgage", "Has Mortgage?"
    )

    cosigner = category_input(
        "HasCoSigner", "Has Co-Signer?"
    )

st.divider()

# -----------------------------------
# 9. Loan Information
# -----------------------------------

st.subheader("3. Loan Information")

col1, col2 = st.columns(2)

with col1:

    loan_amount = numeric_input(
        "LoanAmount", "Loan Amount"
    )

    loan_term = numeric_input(
        "LoanTerm",
        "Loan Term (Months)",
        integer=True
    )

with col2:

    interest_rate = numeric_input(
        "InterestRate",
        "Interest Rate (%)"
    )

    loan_purpose = category_input(
        "LoanPurpose", "Loan Purpose"
    )

st.divider()

# -----------------------------------
# 10. Prediction
# -----------------------------------

if st.button("Predict Loan Default"):

    if income <= 0 or loan_term <= 0:
        st.error(
            "Income and loan term must be positive."
        )
        st.stop()

    # Create borrower input

    input_data = pd.DataFrame([{

        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti_ratio,
        "Education": education,
        "EmploymentType": employment,
        "MaritalStatus": marital_status,
        "HasMortgage": mortgage,
        "HasDependents": dependents,
        "LoanPurpose": loan_purpose,
        "HasCoSigner": cosigner

    }])

    # Feature Engineering

    input_data["LoanToIncomeRatio"] = (
        input_data["LoanAmount"]
        / input_data["Income"]
    )

    input_data["MonthlyIncome"] = (
        input_data["Income"] / 12
    )

    input_data["EstimatedMonthlyPrincipal"] = (
        input_data["LoanAmount"]
        / input_data["LoanTerm"]
    )

    # Match training feature order

    input_data = input_data[
        model_info["features"]
    ]

    # Generate prediction

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    # -----------------------------------
    # 11. Display Results
    # -----------------------------------

    st.subheader("Prediction Results")

    result1, result2 = st.columns(2)

    with result1:

        if prediction == 1:

            st.error(
                "Predicted Class: Default"
            )

        else:

            st.success(
                "Predicted Class: Non-Default"
            )

    with result2:

        st.metric(
            "Estimated Default Risk Score",
            f"{probability:.2%}"
        )

    st.progress(float(probability))

    st.subheader("Financial Summary")

    a, b, c = st.columns(3)

    a.metric(
        "Monthly Income",
        f"{income / 12:,.2f}"
    )

    b.metric(
        "Loan-to-Income Ratio",
        f"{loan_amount / income:.2f}"
    )

    c.metric(
        "Estimated Monthly Principal",
        f"{loan_amount / loan_term:,.2f}"
    )

    st.caption(
        "Estimated monthly principal excludes "
        "interest and other charges."
    )

    st.info(
        "The risk score is an uncalibrated model "
        "estimate, not a verified probability "
        "of future default."
    )

# -----------------------------------
# 12. Footer
# -----------------------------------

st.divider()

st.caption(
    "Loan Default Prediction | "
    "Data Science for Business Applications | HCT"
)

st.caption(
    "Educational purposes only. "
    "Not intended for actual loan approval "
    "or rejection decisions."
)
