import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import re

# =============================
# Load the trained model
# =============================
@st.cache_resource
def load_model():
    """Downloads and loads the trained joblib model from Hugging Face Hub."""
    try:
        # Repository ID and filename for the model
        model_path = hf_hub_download(
            repo_id="Roshanmpraj/Tourism",
            filename="best_tourism_model_v1.joblib",
            repo_type="model"
        )
        model = joblib.load(model_path)
        st.success("Model loaded successfully from Hugging Face Hub!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# =============================
# Utility Function: Column Name Sanitation
# =============================
def to_snake_case(name):
    """Converts PascalCase/CamelCase names to snake_case."""
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower().replace(' ', '_')

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Tourism Prediction", layout="wide")

st.title("🧳 Tourism Package Purchase Prediction")
st.markdown("""
This application predicts whether a customer is likely to purchase a tourism package based on their profile.
---
""")

# Layout for inputs
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.header("Personal Details")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=20000)
        own_car = st.selectbox("Owns a Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        passport = st.selectbox("Has Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")


    with col2:
        st.header("Travel & Pitch Details")
        city_tier = st.selectbox("City Tier", [1, 2, 3], help="Tier of the city the customer is from.")
        num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
        num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
        num_trips = st.number_input("Number of Trips Taken in Last Year", min_value=0, max_value=50, value=5)

        typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        preferred_star = st.selectbox("Preferred Property Star Rating", [1, 2, 3, 4, 5])
        pitch_score = st.slider("Pitch Satisfaction Score (1-5)", 1, 5, 3)
        duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=10)
        num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)


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

# Ensure the DataFrame column names match the model's expected format (snake_case)
input_data.columns = [to_snake_case(col) for col in input_data.columns]


# -----------------------------
# Prediction Button and Logic
# -----------------------------
st.markdown("---")
if st.button("Predict Purchase", type="primary"):
    if model:
        try:
            # Drop the 'other' gender if the model was not trained on it, or handle it as needed
            # Assuming 'Other' will be treated as 'Female' or the most common class for simplicity
            input_data.loc[input_data['gender'] == 'Other', 'gender'] = 'Female'
            
            prediction = model.predict(input_data)[0]
            
            st.subheader("Prediction Result:")
            if prediction == 1:
                st.balloons()
                st.success(f"The model predicts: **✅ HIGH LIKELIHOOD of Purchasing the Package**")
            else:
                st.warning(f"The model predicts: **❌ LOW LIKELIHOOD of Purchasing the Package**")

        except Exception as e:
            st.error(f"An error occurred during prediction. This is often due to missing features or mismatched data types: {e}")
