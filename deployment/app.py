import gradio as gr
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# --------------------------
# Config
# --------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR.parent / "models" / "insurance_model.pkl"
ENCODER_PATH = BASE_DIR.parent / "models" / "encoders.pkl"

# --------------------------
# Load Model & Encoders
# --------------------------
try:
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    print("✅ Model and encoders loaded successfully.")
except Exception as e:
    print("❌ Error loading model or encoders:", e)
    model = None
    encoders = None

# --------------------------
# Define Expected Features
# --------------------------
EXPECTED_FEATURES = [
    "Age", "AnnualIncome", "FamilyMembers", "TourPackageType",
    "Destination", "Season", "TravelInsurance"
]

# --------------------------
# Preprocess Function
# --------------------------
def preprocess_input(data: dict):
    """Converts user input into a model-ready DataFrame."""

    # Fill missing features with default values
    for feature in EXPECTED_FEATURES:
        if feature not in data:
            data[feature] = np.nan

    df = pd.DataFrame([data])

    # Encode categorical columns
    for col, encoder in encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col])
            except Exception:
                # If unseen category appears, replace with 'unknown' or first class
                df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
                df[col] = encoder.transform(df[col])
    return df

# --------------------------
# Prediction Function
# --------------------------
def predict_tourism(age, income, family, package, destination, season, insurance):
    if model is None or encoders is None:
        return "❌ Model or encoders not loaded properly. Please check setup."

    input_data = {
        "Age": age,
        "AnnualIncome": income,
        "FamilyMembers": family,
        "TourPackageType": package,
        "Destination": destination,
        "Season": season,
        "TravelInsurance": insurance
    }

    df = preprocess_input(input_data)
    try:
        prediction = model.predict(df)[0]
        return f"🎯 Predicted result: {prediction}"
    except Exception as e:
        return f"❌ Prediction failed: {str(e)}"

# --------------------------
# Gradio UI
# --------------------------
with gr.Blocks(title="Tourism Prediction App") as app:
    gr.Markdown("## 🧭 Tourism Insurance Prediction App")
    gr.Markdown("Enter your details below to predict your tourism outcome:")

    with gr.Row():
        age = gr.Number(label="Age", value=30)
        income = gr.Number(label="Annual Income", value=50000)
        family = gr.Number(label="Family Members", value=3)

    with gr.Row():
        package = gr.Dropdown(
            label="Tour Package Type",
            choices=["Standard", "Deluxe", "Premium"],
            value="Standard"
        )
        destination = gr.Dropdown(
            label="Destination",
            choices=["Kerala", "Goa", "Kashmir", "Dubai", "Singapore"],
            value="Kerala"
        )
        season = gr.Dropdown(
            label="Season",
            choices=["Summer", "Winter", "Rainy"],
            value="Summer"
        )
        insurance = gr.Dropdown(
            label="Travel Insurance",
            choices=["Yes", "No"],
            value="Yes"
        )

    predict_btn = gr.Button("🚀 Predict")
    output = gr.Textbox(label="Prediction Result")

    predict_btn.click(
        predict_tourism,
        inputs=[age, income, family, package, destination, season, insurance],
        outputs=[output]
    )

# --------------------------
# Launch
# --------------------------
if __name__ == "__main__":
    app.launch()
