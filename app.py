"""Streamlit app for Thermal Control Unit Temperature Predictor
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

MODEL_PATH = "thermal_predictor_model.pkl"

@st.cache_resource
def load_model(path=MODEL_PATH):
    try:
        model = joblib.load(path)
        logging.info("Model loaded from %s", path)
        return model
    except Exception as e:
        logging.exception("Failed to load model")
        st.error(f"Failed to load model: {e}")
        return None


def main():
    st.title("Thermal Control Unit Temperature Predictor")
    st.write("Predict the maximum temperature of a control unit based on input parameters.")

    model = load_model()
    if model is None:
        st.stop()

    # User Inputs
    ambient_temp = st.slider("Ambient Temperature (°C)", 15, 45, 25)
    power_load = st.slider("Power Load (W)", 5, 50, 20)
    cooling_type = st.selectbox("Cooling Type", ["Passive", "Fan", "High-Conductivity"])

    # Map cooling type to numeric
    cooling_map = {"Passive":0, "Fan":1, "High-Conductivity":2}
    cooling_numeric = cooling_map[cooling_type]

    # Predict
    input_data = pd.DataFrame([[ambient_temp, power_load, cooling_numeric]],
                              columns=['AmbientTemp','PowerLoad','CoolingType'])
    predicted_temp = model.predict(input_data)[0]

    st.success(f"Predicted Max Temperature: {predicted_temp:.2f} °C")

    # Simple heatmap visualization (mocked for demo)
    heatmap = np.random.rand(10,10) * (predicted_temp - ambient_temp) + ambient_temp
    fig, ax = plt.subplots()
    cax = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
    ax.set_title("Control Unit Temperature Distribution (Mocked)")
    fig.colorbar(cax)
    st.pyplot(fig)


if __name__ == "__main__":
    main()
