"""Streamlit deployment app for Predictive Maintenance.

Run locally with:
    streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"

st.set_page_config(page_title="Predictive Maintenance", page_icon="🛠️", layout="wide")
st.title("Predictive Maintenance: Machine Failure Risk")
st.write(
    "Enter the current machine operating conditions. The app loads the model "
    "committed in the repository and estimates the probability of machine failure."
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

col1, col2, col3 = st.columns(3)
with col1:
    machine_type = st.selectbox("Machine Type", ["L", "M", "H"], index=0)
    air_temp = st.number_input("Air temperature (K)", 295.0, 305.0, 298.8, 0.1)
with col2:
    process_temp = st.number_input("Process temperature (K)", 305.0, 315.0, 309.8, 0.1)
    speed = st.number_input("Rotational speed (rpm)", 1000, 3000, 1500, 10)
with col3:
    torque = st.number_input("Torque (Nm)", 0.0, 80.0, 40.0, 0.1)
    tool_wear = st.number_input("Tool wear (min)", 0, 260, 100, 1)

input_df = pd.DataFrame([{
    "Type": machine_type,
    "Air_temperature": air_temp,
    "Process_temperature": process_temp,
    "Rotational_speed": speed,
    "Torque": torque,
    "Tool_wear": tool_wear,
}])

st.subheader("Input dataframe")
st.dataframe(input_df, use_container_width=True)

if st.button("Predict failure risk", type="primary"):
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])
    st.metric("Estimated failure probability", f"{probability:.1%}")
    if prediction == 1:
        st.error("High failure risk predicted. Prioritise inspection or maintenance.")
    else:
        st.success("Low failure risk predicted. Continue monitoring.")

st.caption(
    "Model: tuned scikit-learn pipeline saved as models/best_model.joblib. "
    "Dataset source: UCI AI4I 2020 Predictive Maintenance Dataset."
)
