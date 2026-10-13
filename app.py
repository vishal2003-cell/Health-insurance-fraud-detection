
import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="HealthShield | Fraud Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #f1f5f9;
}
.block-container {
    padding-top: 2rem;
    max-width: 1250px;
}
[data-testid="stMetric"] {
    background: #172238;
    border: 1px solid #293b58;
    padding: 18px;
    border-radius: 12px;
}
div.stButton > button {
    width: 100%;
    border-radius: 9px;
    height: 3rem;
    font-weight: 600;
}
section[data-testid="stSidebar"] {
    background-color: #111b2e;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL FILES ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("logisticRegression.pkl")
    scaler = joblib.load("scaler.pkl")
    model_columns = list(joblib.load("columns.pkl"))
    return model, scaler, model_columns


try:
    model, scaler, model_columns = load_artifacts()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.info(
        "Check that logisticRegression.pkl, scaler.pkl "
        "and columns.pkl exist in your project folder."
    )
    st.stop()


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🏥 HealthShield")
    st.caption("Machine Learning Project")
    st.markdown("---")

    st.subheader("About the Project")
    st.write(
        "This application uses a trained Machine Learning "
        "model to estimate whether an insurance claim "
        "may be fraudulent."
    )

    st.markdown("---")
    st.caption("Model: Logistic Regression")
    st.caption("Interface: Streamlit")
    st.caption("Output: Risk prediction")

    st.warning(
        "Predictions are estimates and should not be "
        "used as the sole basis for claim decisions."
    )


# ---------------- HEADER ----------------
st.title("🏥 Health Insurance Fraud Detection")
st.write(
    "Enter the claim details below to estimate "
    "the likelihood of insurance fraud."
)

st.markdown("---")

# ---------------- INPUT FORM ----------------
with st.form("claim_form"):

    st.subheader("👤 Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age (years)", 1, 120, 30
        )

    with col2:
        gender = st.selectbox(
            "Gender", ["Male", "Female"]
        )

    with col3:
        emergency = st.selectbox(
            "Emergency Claim?", ["Yes", "No"]
        )

    st.markdown("---")
    st.subheader("🏥 Hospital & Treatment")

    col1, col2, col3 = st.columns(3)

    with col1:
        hospital_type = st.selectbox(
            "Hospital Type",
            ["Private", "Trust", "Government"]
        )

    with col2:
        treatment_cost = st.number_input(
            "Treatment Cost (INR)",
            min_value=0,
            value=40000,
            step=1000
        )

    with col3:
        days_hospitalized = st.number_input(
            "Days Hospitalized",
            min_value=0,
            max_value=365,
            value=3
        )

    st.markdown("---")
    st.subheader("📋 Insurance Claim Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        claim_amount = st.number_input(
            "Claim Amount (INR)",
            min_value=0,
            value=50000,
            step=1000
        )

    with col2:
        previous_claims = st.number_input(
            "Previous Claims",
            min_value=0,
            max_value=20,
            value=0
        )

    with col3:
        policy_years = st.number_input(
            "Policy Holding Years",
            min_value=0,
            max_value=50,
            value=2
        )

    submitted = st.form_submit_button(
        "🔍 Predict Fraud Status",
        type="primary",
        use_container_width=True
    )


# ---------------- PREDICTION ----------------
if submitted:

    if claim_amount == 0:
        st.warning(
            "Claim amount is zero. Please verify "
            "the entered claim details."
        )

    try:
        is_male = 1 if gender == "Male" else 0
        emergency_val = 1 if emergency == "Yes" else 0

        input_data = {
            "Age": age,
            "is_male": is_male,
            "Hospital_Type": hospital_type,
            "Claim_Amount": claim_amount,
            "Treatment_Cost": treatment_cost,
            "Previous_Claims": previous_claims,
            "Policy_Years": policy_years,
            "Days_Hospitalized": days_hospitalized,
            "Emergency": emergency_val
        }

        input_df = pd.DataFrame([input_data])

        # Keep the same encoding approach as the original app
        input_encoded = pd.get_dummies(
            input_df,
            columns=["Hospital_Type"],
            drop_first=True
        )

        # Match the exact feature names and order
        final_features = input_encoded.reindex(
            columns=model_columns,
            fill_value=0
        )

        numeric_cols = [
            "Age",
            "Claim_Amount",
            "Treatment_Cost",
            "Previous_Claims",
            "Policy_Years",
            "Days_Hospitalized"
        ]

        # Scale the same numeric features as during training
        final_features[numeric_cols] = scaler.transform(
            final_features[numeric_cols]
        )

        prediction = model.predict(final_features)[0]
        probabilities = model.predict_proba(final_features)[0]

        # Assumes class 1 = Fraud and class 0 = Legitimate
        classes = list(model.classes_)
        fraud_index = classes.index(1)
        genuine_index = classes.index(0)

        fraud_probability = probabilities[fraud_index] * 100
        genuine_probability = probabilities[genuine_index] * 100

        st.markdown("---")
        st.header("📊 Prediction Results")

        result_col, probability_col = st.columns(2)

        with result_col:
            if prediction == 1:
                st.error("🚨 Potential Fraud Detected")
                st.write(
                    "The model classified this claim "
                    "as potentially fraudulent."
                )
            else:
                st.success("✅ Likely Legitimate")
                st.write(
                    "The model classified this claim "
                    "as likely legitimate."
                )

        with probability_col:
            st.metric(
                "Estimated Fraud Probability",
                f"{fraud_probability:.2f}%"
            )
            st.progress(
                int(round(fraud_probability)),
                text="Estimated fraud probability"
            )

        st.caption(
            "Probability is a model estimate, not proof of fraud "
            "or a guarantee of legitimacy."
        )

        st.markdown("---")
        st.subheader("📌 Claim Summary")

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Claim Amount",
            f"₹{claim_amount:,.0f}"
        )
        m2.metric(
            "Treatment Cost",
            f"₹{treatment_cost:,.0f}"
        )
        m3.metric(
            "Previous Claims",
            str(previous_claims)
        )

        st.subheader("💰 Cost Comparison")

        cost_chart = pd.DataFrame({
            "Category": ["Claim Amount", "Treatment Cost"],
            "Amount (INR)": [claim_amount, treatment_cost]
        }).set_index("Category")

        st.bar_chart(cost_chart)

        with st.expander("View detailed prediction probabilities"):
            st.write(
                f"**Potential fraud:** {fraud_probability:.2f}%"
            )
            st.write(
                f"**Likely legitimate:** {genuine_probability:.2f}%"
            )

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info(
            "Verify that your saved model, scaler, column names "
            "and input preprocessing match the training notebook."
        )


# ---------------- FOOTER ----------------
st.markdown("---")
st.caption(
    "HealthShield | Health Insurance Fraud Detection "
    "using Machine Learning"
)
