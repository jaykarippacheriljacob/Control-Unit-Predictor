"""Streamlit app for Thermal Control Unit Temperature Predictor
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp
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

    # Physics-inspired radial heatmap
    grid_size = 50
    x, y = np.meshgrid(np.linspace(-1, 1, grid_size),
                        np.linspace(-1, 1, grid_size))
    r = np.sqrt(x**2 + y**2)
    r = r / r.max()

    heatmap = ambient_temp + (predicted_temp - ambient_temp) * np.exp(-3 * r)

    # Create subplots with 3D surface and contour plot
    fig = sp.make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'contour'}]]
    )

    # 3D Surface plot
    fig.add_trace(
        go.Surface(x=x[0], y=y[:, 0], z=heatmap, colorscale='Hot',
                   name='Temperature'),
        row=1, col=1
    )

    # Contour plot
    fig.add_trace(
        go.Contour(x=x[0], y=y[:, 0], z=heatmap, colorscale='Hot',
                   showscale=True, colorbar=dict(x=1.02)),
        row=1, col=2
    )

    fig.update_layout(height=600, title_text="Control Unit Temperature Distribution",
                      showlegend=False)
    fig.update_xaxes(title_text="Width", row=1, col=2)
    fig.update_yaxes(title_text="Height", row=1, col=2)

    st.plotly_chart(fig, width='stretch')


if __name__ == "__main__":
    main()
