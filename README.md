# Walmart-Sales-Retail-analysis
# Walmart Sales Forecasting (Databricks + ML)

## 📌 Overview
This project implements an end-to-end data engineering and machine learning pipeline on the Walmart dataset:
- Built a **Bronze → Silver → Gold** data pipeline in Databricks
- Applied data cleaning (handling NA values, type casting, duplicates)
- Performed feature engineering (lag features, rolling averages, holiday interactions)
- Compared multiple forecasting models:
  - RandomForestRegressor
  - XGBoost
  - Prophet
  - LSTM (deep learning)

## 📂 Project Structure
- `notebooks/` → Databricks notebooks
- `scripts/` → Python ETL & ML scripts
- `reports/` → Generated plots, Prophet visualizations, evaluation metrics
- `requirements.txt` → Dependencies

## ⚙️ Tech Stack
- **Data Engineering**: PySpark, Databricks (Bronze-Silver-Gold architecture)
- **ML**: scikit-learn, XGBoost, Prophet, TensorFlow/Keras
- **Visualization**: Matplotlib, Seaborn

## 📊 Results
| Model        | RMSE   | R²   |
|--------------|--------|------|
| RandomForest | xx     | xx   |
| XGBoost      | xx     | xx   |
| Prophet      | xx     | xx   |
| LSTM         | xx     | xx   |

LSTM and Prophet performed best for forecasting weekly sales trends.

## 🚀 How to Run
1. Clone this repo  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

---

## 5. Push to GitHub
```bash
# Initialize repo
git init
git add .
git commit -m "Initial commit: Walmart Sales Forecasting Project"

# Add GitHub remote
git branch -M main
git remote add origin https://github.com/<your-username>/walmart-sales-forecasting.git
git push -u origin main
