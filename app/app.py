"""
Streamlit Web Application for End-to-End Model Inference.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="End-to-End ML Platform",
    page_icon="⚡",
    layout="wide",
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model_pipeline.joblib"


@st.cache_resource
def load_model_artifact(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


st.title("⚡ Environmental Prediction Dashboard")
st.markdown(
    "Production-grade End-to-End machine learning application. "
    "Adjust environmental features in the sidebar to run inference in real time."
)

artifact = load_model_artifact(MODEL_PATH)

if artifact is None:
    st.error(
        f"**Artifact Not Found:** Model pipeline is missing at `{MODEL_PATH}`.\n\n"
        "Please train the model first by running:\n"
        "```bash\npython src/train.py\n```"
    )
    st.stop()

pipeline = artifact["pipeline"]
features = artifact["feature_names"]
metrics = artifact["metrics"]

# Sidebar: Model Performance Indicators
st.sidebar.header("📊 Model Performance (Test Set)")
st.sidebar.metric("R² Score", f"{metrics['r2']}")
st.sidebar.metric("RMSE", f"{metrics['rmse']}")
st.sidebar.metric("MAE", f"{metrics['mae']}")
st.sidebar.divider()

# Sidebar: User Input Controls
st.sidebar.header("🔧 Input Parameters")
temperature = st.sidebar.slider("Ambient Temperature (°C)", -5.0, 35.0, 18.0, step=0.5)
humidity = st.sidebar.slider("Relative Humidity (%)", 20.0, 95.0, 55.0, step=1.0)
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0.5, 20.0, 6.5, step=0.5)
traffic_density = st.sidebar.slider("Traffic Density Index (1-100)", 10.0, 100.0, 65.0, step=1.0)
hour = st.sidebar.selectbox("Hour of Day", options=list(range(24)), index=14)

# Main Panel: Prediction and Explanations
col1, col2 = st.columns([1, 1])

input_data = pd.DataFrame(
    [
        {
            "temperature_c": temperature,
            "humidity_pct": humidity,
            "wind_speed_kmh": wind_speed,
            "traffic_density": traffic_density,
            "hour_of_day": hour,
        }
    ]
)

with col1:
    st.subheader("Current Inference Vector")
    st.dataframe(input_data, use_container_width=True)

    if st.button("Run Inference", type="primary"):
        prediction = pipeline.predict(input_data)[0]
        prediction_val = round(float(prediction), 1)

        with col2:
            st.subheader("Inference Result")
            if prediction_val < 30:
                status_color = "normal"
                status_label = "Good / Low Pollution"
            elif prediction_val < 60:
                status_color = "off"
                status_label = "Moderate"
            else:
                status_color = "inverse"
                status_label = "High / Unfavorable"

            st.metric(
                label="Predicted Target (AQI)",
                value=f"{prediction_val} µg/m³",
                delta=status_label,
                delta_color=status_color,
            )

            st.progress(min(int(prediction_val), 100) / 100)
            st.caption(
                "Prediction powered by scikit-learn RandomForestRegressor pipeline with standardized input scaling."
            )
