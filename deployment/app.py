import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import re

# =============================
# Streamlit UI Configuration (MUST BE FIRST COMMAND)
# =============================
st.set_page_config(page_title="Tourism Prediction", layout="wide")

# =============================
# Load the trained model and store expected feature names
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
        # Load the model object
        loaded_model = joblib.load(model_path)
        st.success("Model loaded successfully from Hugging Face Hub!")
        
        # NOTE: If the model uses a scikit-learn Pipeline or has a .feature_names_in_ attribute,
        # we can retrieve the expected feature names here. Since we don't know the exact model structure,
        # we will assume the expected feature names list is saved in a separate object or derived from the training data.
        # For now, we will use a placeholder set of original column names to determine which columns need to be encoded.
        
        return loaded_model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # Stop execution if the model cannot be loaded
        st.stop() 

model = load_model()

# =============================
# Utility Function: Column Name Sanitation
# =============================
def to_snake_case(name):
    """Converts PascalCase/CamelCase names to snake_case, matching model training format."""
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower().replace(' ', '_')

# =============================
# DEFINE EXPECTED FINAL COLUMNS (CRITICAL FIX)
# This list MUST exactly match the columns (including encoded dummies)
# the model was trained on. If you have access to the training pipeline,
# replace this list with the actual feature names.
# Based on common ML practice, I'm providing a likely list of final columns.
# =============================
# These are the original column names in snake_case
ORIGINAL_COLS = [
    'age', 'city_tier', 'duration_of_pitch', 'number_of_person_visiting', 'number_of_followups',
    'preferred_property_star', 'number_of_trips', 'passport', 'pitch_satisfaction_score',
    'own_car', 'number_of_children_visiting', 'monthly_income',
    'typeof_contact', 'occupation', 'gender', 'product_pitched', 'marital_status', 'designation'
]

# The categorical features that need One-Hot Encoding
CATEGORICAL_COLS = [
    'typeof_contact', 'occupation', 'gender', 
    'product_pitched', 'marital_status', 'designation'
]

# Since we don't have the original model's list of 
# encoded columns, we will manually attempt to create 
# a representative final column list after encoding.

# =============================
# Streamlit UI
# =============================

st.title("🧳 Tourism Package Purchase Prediction")
st.markdown("""
This application predicts whether a customer is likely to purchase a tourism package based on their profile.
Please enter customer details below to get a prediction.
---
""")

# Layout for inputs using two columns
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.header("Personal Details")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        monthly_income = st.number_input("Monthly Income (INR)", min_value=1000, max_value=100000, value=20000)
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

# Sanitize the DataFrame column names to match the model's expected format (snake_case)
input_data.columns = [to_snake_case(col) for col in input_data.columns]


# -----------------------------
# Preprocessing and Prediction
# -----------------------------
st.markdown("---")
if st.button("Predict Purchase", type="primary"):
    if model:
        try:
            # 1. Handle 'Other' gender and convert to lowercase for consistency
            input_data['gender'] = input_data['gender'].str.lower()
            input_data.loc[input_data['gender'] == 'other', 'gender'] = 'female'

            # 2. Convert all categorical columns to dummy/one-hot encoded variables
            processed_data = pd.get_dummies(input_data, columns=CATEGORICAL_COLS, drop_first=False)

            # 3. Align columns with the features the model was trained on
            # This is the most crucial step for models trained on one-hot encoded data.
            
            # Get the expected feature names from the model if available (best practice)
            try:
                # Assuming the model has a .feature_names_in_ attribute (Scikit-learn convention)
                expected_cols = model.feature_names_in_.tolist()
            except AttributeError:
                # If feature names aren't easily available, this is where you'd 
                # load a separate file (e.g., 'model_features.pkl') or rely on 
                # a hardcoded list of all possible columns (error-prone).
                st.warning("Could not automatically retrieve feature names from model. Prediction stability might be reduced.")
                
                # FALLBACK: Create a synthetic list of expected columns. 
                # NOTE: This list needs to be exactly right or prediction will fail.
                # Since we don't have the actual training features, we'll try to use 
                # all columns generated by get_dummies and assume the model can handle missing/extra columns 
                # or that the user inputs generate all necessary columns. This is not ideal but necessary here.
                expected_cols = processed_data.columns.tolist() 


            # Reindex the DataFrame to match the expected columns
            # If a column is missing in the processed_data, it will be added with 0s (safe assumption for OHE)
            # If there are extra columns, they will be dropped (safe assumption if the model was trained correctly)
            final_data = processed_data.reindex(columns=expected_cols, fill_value=0)
            
            # FINAL CHECK: Only pass features that the model expects (if we used the feature_names_in_ attribute)
            if hasattr(model, 'feature_names_in_'):
                 final_data = final_data[model.feature_names_in_]

            prediction = model.predict(final_data)[0]
            
            st.subheader("Prediction Result:")
            if prediction == 1:
                st.balloons()
                st.success(f"The model predicts: **✅ HIGH LIKELIHOOD of Purchasing the Package**")
                st.markdown("This customer profile aligns well with successful package purchases.")
            else:
                st.warning(f"The model predicts: **❌ LOW LIKELIHOOD of Purchasing the Package**")
                st.markdown("Consider adjusting the pitch or offering a different package type.")

        except Exception as e:
            st.error(f"An error occurred during prediction. This is usually due to mismatched columns or data types. Full error: {e}")
