import streamlit as st 
import pandas as pd
import joblib

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

# Load model and encoders
model = joblib.load("best_extra_trees_model.pkl")
encoders = {col: joblib.load(f"{col}_label_encoder.pkl") for col in ["Sex", "Housing", "Saving accounts", "Checking account"]}

# Header
st.title("💳 Credit Risk Predictor")
st.markdown("---")
st.markdown("### Enter applicant details to predict credit risk")

# Create form container
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**👤 Personal Information**")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        sex = st.selectbox("Sex", ["male", "female"])
        job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
        housing = st.selectbox("Housing", ["own", "rent", "free"])
    
    with col2:
        st.markdown("**💰 Financial Information**")
        credit_amount = st.number_input("Credit Amount (DM)", min_value=0, value=1000, step=100)
        duration = st.number_input("Duration (months)", min_value=1, value=12)
        saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich"])
        checking_account = st.selectbox("Checking Account", ["little", "moderate"])

st.markdown("---")

# Prepare input data
input_data = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job],       
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration]
})

# Predict button
if st.button("🔮 Predict Credit Risk"):
    pred = model.predict(input_data)[0]
    
    st.markdown("---")
    
    # Results with icons
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

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #6c757d;'>Model: Extra Trees Classifier | Accuracy: 64.76%</p>", unsafe_allow_html=True)
