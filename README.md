# End-to-End Data Science Project Template
## 📁 Repository Structure

```text
├── data/
│   ├── raw/               <- Immutable raw data source files
│   └── processed/         <- Cleaned, transformed data ready for modeling
├── models/                <- Serialized model pipelines and artifacts
├── notebooks/             <- Jupyter notebooks for EDA and experimentation
├── src/
│   ├── __init__.py
│   └── train.py           <- Automated training and evaluation pipeline
├── app/
│   └── app.py             <- Streamlit web dashboard for live inference
├── .gitignore             <- Pre-configured Git exclusion rules
├── requirements.txt       <- Python dependencies
└── README.md
```
# 🚀 Quickstart Guide
## 1. Clone the Repository
```Bash
git clone https://github.com/Olat1337/e2e-data-science-template.git
cd e2e-data-science-template
```
## 2. Set Up a Virtual Environment
macOS / Linux:
```Bash
python3 -m venv .venv
source .venv/bin/activate
```
Windows (PowerShell):
```Bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```
(If PowerShell execution policy blocks the script, run Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass).
## 3. Install Dependencies
```Bash
pip install --upgrade pip
pip install -r requirements.txt
```
## 4. Train the Model Pipeline
Run the automated data preparation and model training script:
```Bash
python src/train.py
```
This script will:
Verify or generate sample training data in data/raw/.
Fit a preprocessing and RandomForestRegressor pipeline.
Print validation metrics (R^2, MAE, RMSE).
Export the artifact to models/model_pipeline.joblib.
##5. Launch the Web Application
```Bash
streamlit run app/app.py
```
The application will open in your browser at http://localhost:8501.
# 🛠️ Customizing for Your Team Project
- Replace the Data: Place your CSV/Parquet files in data/raw/ or connect an API scraper in src/.
- Experiment in Notebooks: Use notebooks/ to clean data, engineer features, and compare models.
- Automate in src/train.py: Move your best data preparation steps and modeling code from notebooks into a reproducible Python pipeline.
- Build the UI in app/app.py: Customize input widgets and result visualizations to match your project domain.
