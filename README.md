# TechSight Business Performance Dashboard 📊

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)

An interactive, machine-learning-powered business performance dashboard built for **TechSight ICT Solutions**. This application ingests synthesized (or real) operational data to generate highly intuitive KPIs, predictive sales forecasting, and active inventory alerting systems.

## ✨ Core Features
*   **KPI Tracking Engine**: Computes exact operational metrics on sales, service times, and retention natively.
*   **Predictive Sales Forecaster (ML)**: Automatically projects upcoming monthly revenue against fixed $5,000 monthly targets using scikit-learn models.
*   **Churn Classifier (ML)**: Dynamically maps customer activity into logistic regression predictors flagging individual `High`, `Medium`, and `Low` risk CRM states.
*   **Inventory Alerts**: Rolling historical analyses tracking hardware drops beneath reorder levels.
*   **Hot-swappable Custom Data**: Users can drag-and-drop their actual CRM and POS `.csv` exports directly mapping to live charts dynamically.

## 📁 Repository Structure
```text
techsight_dashboard/
├── data_processor.py      # Core pandas-driven data munging, cleansing, and aggregations
├── ml_models.py           # Scikit-learn extensions for predictive logic
├── dashboard.py           # The robust Streamlit frontend UI
├── generate_data.py       # Deterministic generator simulating $38.4k target synthetic environments
├── requirements.txt       # Environment dependencies
└── tests/
    └── test_processor.py  # Validation parameters guaranteeing 100% KPI precision 
```

## 🚀 Running Locally

1. **Install Python environment dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Execute the generator (Optional; pre-loaded):**
   ```bash
   python generate_data.py
   ```
3. **Launch the Streamlit UI locally:**
   ```bash
   python -m streamlit run dashboard.py
   ```

## 🌐 Deploying to Streamlit Community Cloud
Deploying is 1-click native:
1. Log into [Streamlit Share](https://share.streamlit.io).
2. Point a new deployment to your `main` branch.
3. Select `dashboard.py` as the Primary File.

## 🧪 Testing
We maintain full operational testing matrices validating that the pipeline accurately models exact numerical definitions.
Execute our tests natively via:
```bash
pytest tests/test_processor.py
```
