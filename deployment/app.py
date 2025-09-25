import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import re

# =============================
# Load the trained model
# =============================
try:
    # Replace with your repo id where model is uploaded
    model_path = hf_hub_download(
        repo_id="Roshanmpraj/Tourism",
        filename="best_tourism_model_v1.joblib",
        repo_type="model"
    )
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# =============================
# Streamlit UI
# =============================
st.title("🧳 Tourism Prediction App")
st.write("""
This application predicts whether a customer is likely to purchase a tourism package.
Please enter customer details below to get a prediction.
""")

# -----------------------------
# User Inputs
# -----------------------------
# Numeric features
age = st.number_input("Age", min_value=18, max_value=100, value=30)
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_pitch = st.number_input("Duration of Pitch", min_value=0, max_value=60, value=10)
num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)
preferred_star = st.selectbox("Preferred Property Star Rating", [1, 2, 3, 4, 5])
num_trips = st.number_input("Number of Trips", min_value=0, max_value=50, value=5)
passport = st.selectbox("Has Passport", [0, 1])
pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
own_car = st.selectbox("Owns a Car", [0, 1])
num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=20000)

# Categorical features
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

# Assemble input data into DataFrame
input_data = pd.DataFrame([{
    "Age": age,
    "CityTier": city_tier,
    "DurationOfPitch": duration_pitch,
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": num_followups,
    "PreferredPropertyStar": preferred_star,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "MonthlyIncome": monthly_income,
    "TypeofContact": typeof_contact,
    "Occupation": occupation,
    "Gender": gender,
    "ProductPitched": product_pitched,
    "MaritalStatus": marital_status,
    "Designation": designation
}])

# =============================
# FIX: Sanitize column names
# The model was likely trained on lowercase, snake_case column names.
# This ensures the input DataFrame matches the model's expectations.
# =============================
def to_snake_case(name):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower().replace(' ', '_')

input_data.columns = [to_snake_case(col) for col in input_data.columns]


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Purchase"):
    try:
        prediction = model.predict(input_data)[0]
        result = "✅ Will Purchase Package" if prediction == 1 else "❌ Will Not Purchase Package"
        st.subheader("Prediction Result:")
        st.success(f"The model predicts: **{result}**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
