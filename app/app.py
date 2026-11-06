"""
Streamlit Web Application for End-to-End Model Inference.
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Hardware Valuation ML Platform",
    page_icon="💻",
    layout="wide",
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model_pipeline.joblib"


@st.cache_resource
def load_artifact(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


st.title("💻 Hardware Valuation Platform (End-to-End Demo)")
st.markdown(
    "Reference architecture template for machine learning workshops. "
    "Select technical specifications in the sidebar to estimate device market value in real time."
)

artifact = load_artifact(MODEL_PATH)

if artifact is None:
    st.error(
        f"**Model Artifact Missing:** Could not locate `{MODEL_PATH}`.\n\n"
        "Train the model first using:\n"
        "```bash\npython src/train.py\n```"
    )
    st.stop()

pipeline = artifact["pipeline"]
metrics = artifact["metrics"]

# Sidebar - Metrics Display
st.sidebar.header("📊 Model Metrics (Test Set)")
st.sidebar.metric("R² Score", f"{metrics['r2']}")
st.sidebar.metric("RMSE", f"${metrics['rmse']}")
st.sidebar.metric("MAE", f"${metrics['mae']}")
st.sidebar.divider()

# Sidebar - Input Features
st.sidebar.header("⚙️ Specifications")
ram_gb = st.sidebar.select_slider("RAM (GB)", options=[8, 16, 32, 64], value=16)
cpu_cores = st.sidebar.select_slider("CPU Cores", options=[4, 6, 8, 12, 16], value=8)
storage_gb = st.sidebar.select_slider("SSD Storage (GB)", options=[256, 512, 1024, 2048], value=512)
weight_kg = st.sidebar.slider("Weight (kg)", min_value=1.0, max_value=3.0, value=1.6, step=0.1)
gpu_tier = st.sidebar.checkbox("Dedicated Discrete GPU", value=False)

# Main Section - Inference
col1, col2 = st.columns([1, 1])

input_vector = pd.DataFrame(
    [
        {
            "ram_gb": ram_gb,
            "cpu_cores": cpu_cores,
            "storage_gb": storage_gb,
            "weight_kg": weight_kg,
            "has_dedicated_gpu": 1 if gpu_tier else 0,
        }
    ]
)

with col1:
    st.subheader("Configured Hardware Profile")
    st.dataframe(input_vector, use_container_width=True)

    if st.button("Calculate Valuation", type="primary"):
        prediction = pipeline.predict(input_vector)[0]
        estimated_price = round(float(prediction), 2)

        with col2:
            st.subheader("Predicted Market Value")
            st.metric(
                label="Estimated Price (USD)",
                value=f"${estimated_price:,.2f}",
                delta="Normal Market Range",
            )
            st.info(
                "Prediction generated via scikit-learn pipeline (StandardScaler + RandomForestRegressor)."
            )
