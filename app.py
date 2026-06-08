import os

import requests
import streamlit as st

# Point this at the deployed API. Override via Streamlit secrets or env var.
API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://localhost:8000"))

# Page config
st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="centered")

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 12px;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .css-1d3km3w {
        padding: 2rem;
    }
    .stSelectbox, .stNumberInput {
        background-color: white;
        border-radius: 8px;
        padding: 8px;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        padding-bottom: 20px;
    }
    .stSuccess {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 20px;
    }
    .stError {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 10px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=30)
def api_healthy() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


# Header
st.title("💳 Credit Risk Predictor")
st.markdown("---")
st.markdown("### Enter applicant details to predict credit risk")

# Create form container
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**👤 Personal Information**")

        st.markdown("**Age**")
        age = st.number_input("Age", min_value=18, max_value=100, value=30, label_visibility="collapsed")

        st.markdown("**Sex**")
        sex = st.selectbox("Sex", ["male", "female"], label_visibility="collapsed")

        st.markdown("**Job (0-3)**")
        job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1, label_visibility="collapsed")

        st.markdown("**Housing**")
        housing = st.selectbox("Housing", ["own", "rent", "free"], label_visibility="collapsed")

    with col2:
        st.markdown("**💰 Financial Information**")

        st.markdown("**Credit Amount (DM)**")
        credit_amount = st.number_input("Credit Amount (DM)", min_value=0, value=1000, step=100, label_visibility="collapsed")

        st.markdown("**Duration (months)**")
        duration = st.number_input("Duration (months)", min_value=1, value=12, label_visibility="collapsed")

        st.markdown("**Saving Accounts**")
        saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich"], label_visibility="collapsed")

        st.markdown("**Checking Account**")
        checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich"], label_visibility="collapsed")

        st.markdown("**Purpose**")
        purpose = st.selectbox(
            "Purpose",
            [
                "car",
                "radio/TV",
                "furniture/equipment",
                "business",
                "education",
                "repairs",
                "domestic appliances",
                "vacation/others",
            ],
            label_visibility="collapsed",
        )

st.markdown("---")

# Build payload using the exact field names the API expects (Pydantic aliases)
payload = {
    "Age": age,
    "Sex": sex,
    "Job": job,
    "Purpose": purpose,
    "Housing": housing,
    "Saving accounts": saving_accounts,
    "Checking account": checking_account,
    "Credit amount": credit_amount,
    "Duration": duration,
}

# Predict button
if st.button("🔮 Predict Credit Risk"):
    if not api_healthy():
        st.error(f"API not reachable at {API_URL}. Check that the service is running.")
    else:
        try:
            with st.spinner("Scoring..."):
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
            resp.raise_for_status()
            body = resp.json()
            pred = body["prediction"]
            proba_bad = body.get("probability_bad_risk")

            st.markdown("---")

            if pred == 1:
                st.markdown("""
                <div style='text-align: center; padding: 20px; background-color: #d4edda; border-radius: 10px; border: 2px solid #28a745;'>
                    <h2 style='color: #28a745; margin: 0;'>✅ Good Credit Risk</h2>
                    <p style='font-size: 18px; color: #28a745;'>The applicant is likely to repay the credit.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='text-align: center; padding: 20px; background-color: #f8d7da; border-radius: 10px; border: 2px solid #dc3545;'>
                    <h2 style='color: #dc3545; margin: 0;'>❌ Bad Credit Risk</h2>
                    <p style='font-size: 18px; color: #dc3545;'>The applicant may default on the credit.</p>
                </div>
                """, unsafe_allow_html=True)

            if proba_bad is not None:
                st.caption(f"Probability of bad risk: {proba_bad:.1%}")

        except requests.HTTPError:
            st.error(f"Prediction failed ({resp.status_code}): {resp.text}")
        except requests.RequestException as exc:
            st.error(f"Request error: {exc}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #6c757d;'>Model: Extra Trees Classifier </p>", unsafe_allow_html=True)
