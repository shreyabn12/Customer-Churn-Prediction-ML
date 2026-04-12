# Customer Churn Prediction System

An end-to-end, production-grade machine learning system that predicts customer churn using the IBM Telco Customer Churn dataset. Built with a focus on clean, modular code, model explainability, and real-world deployment practices.

---

## Live Demo

> API and Dashboard links will be added after Phase 7 (Deployment)

---

## Project Overview

Customer churn — when a customer stops using a service — is one of the most costly problems in subscription businesses. This project builds a complete ML pipeline that not only predicts which customers are likely to churn, but also explains *why* the model makes each prediction, making it actionable for business teams.

The system achieves an **AUC-ROC of 0.8535** on the test set using XGBoost with Optuna hyperparameter tuning and SMOTE for class imbalance correction.

---

## Architecture

---

## Dataset

**IBM Telco Customer Churn** — sourced from [Kaggle](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset/data)

The dataset contains 7,043 customer records with 33 columns covering demographics, account information, and subscribed services. The target variable is `Churn Label` (Yes/No), with a natural class imbalance of approximately 73.5% No / 26.5% Yes.

Columns identified as **data leakage** and excluded from training: `Churn Value`, `Churn Score`, `Churn Reason`, `CLTV`. These columns are derived from or recorded after the churn event, so using them would give the model unfair information it wouldn't have in a real prediction scenario.

---

## Project Structure

---

## Models Trained

All four models were trained using a consistent pipeline: ColumnTransformer preprocessing → SMOTE balancing → classifier. Hyperparameters were tuned using Optuna (30 trials, TPE sampler) with 5-fold stratified cross-validation.

| Model | AUC-ROC | F1 Score | Precision | Recall |
|---|---|---|---|---|
| **XGBoost** ✅ | **0.8535** | **0.6355** | 0.5761 | 0.7086 |
| LightGBM | 0.8535 | 0.6333 | 0.5833 | 0.6925 |
| Random Forest | 0.8510 | 0.6309 | 0.5509 | 0.7380 |
| Logistic Regression | 0.8462 | 0.6149 | 0.5106 | 0.7727 |

XGBoost was selected as the best model based on AUC-ROC. Notably, Logistic Regression achieves the highest recall (0.7727), meaning it catches the most actual churners — which may be preferable in a business context where missing a churner is more costly than a false alarm.

---

## Key Technical Decisions

**Why SMOTE inside the Pipeline?** SMOTE is placed as a pipeline step (using `imblearn.pipeline.Pipeline`) rather than applied to the data directly. This ensures synthetic samples are only created from training folds during cross-validation, never leaking into validation or test sets.

**Why AUC-ROC over Accuracy?** With a 73.5/26.5 class split, a model that always predicts "No Churn" achieves 73.5% accuracy while being completely useless. AUC-ROC is threshold-independent and measures how well the model separates the two classes regardless of the operating point.

**Why drop CLTV and Churn Score?** These are IBM-computed scores that incorporate churn probability. Including them would make the model appear extremely accurate but would be meaningless in production, where those scores are not available at prediction time.

---

## Feature Engineering

Two new features were created from the raw columns to improve model performance.

`charges_per_month` is computed as `total_charges / tenure_months` and represents a customer's average monthly spend. New customers (tenure = 0) use `monthly_charges` directly to avoid division by zero. This feature captures spending consistency better than either raw column alone.

`tenure_group_encoded` buckets raw tenure into four ordinal groups: new (0–12 months), developing (13–24), established (25–48), and loyal (49+). This helps the model capture non-linear tenure effects — the difference in churn risk between 1 month and 12 months of tenure is much larger than between 50 and 60 months.

---

## SHAP Explainability

Model predictions are explained using SHAP (SHapley Additive exPlanations), which fairly attributes each feature's contribution to any individual prediction. Three visualization types are produced.

The **summary plot** shows global feature importance across all test customers — which features move predictions the most and in which direction. The **waterfall plot** explains a single high-risk customer's prediction, showing which specific factors drove their churn probability above the threshold. The **dependence plot** shows how churn risk varies with tenure, revealing that the relationship is non-linear and strongest for customers under 12 months old.

---

## How to Run Locally

**Clone the repository:**
```bash
git clone https://github.com/shreyabn12/Customer-Churn-Prediction-ML.git
cd churn-prediction
```

**Create environment and install dependencies:**
```bash
python -m venv venv
# Mac/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate.bat

pip install -r requirements.txt
```

**Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset/data), rename it to `Telco_customer_churn.xlsx`, and place it in `data/raw/`.

**Run exploratory analysis:**
```bash
jupyter notebook notebooks/01_eda.ipynb
```

**Train all models:**
```bash
python src/train.py
```

**View MLflow experiment dashboard:**
```bash
mlflow ui
# Open http://127.0.0.1:5000 in your browser
```

---

## Phases Completed

- ✅ Phase 1 — Project structure, virtual environment, data loading
- ✅ Phase 2 — Feature engineering and preprocessing pipeline
- ✅ Phase 3 — Model training (4 models), Optuna tuning, SMOTE, MLflow tracking
- ✅ Phase 4 — SHAP explainability (summary, waterfall, dependence plots)
- 🔲 Phase 5 — FastAPI REST endpoint
- 🔲 Phase 6 — Streamlit interactive dashboard
- 🔲 Phase 7 — Docker containerization
- 🔲 Phase 8 — Deployment (Render / Hugging Face Spaces)
- 🔲 Phase 9 — Evidently AI data drift monitoring

---

## Tech Stack

Python 3.11 · pandas · scikit-learn · XGBoost · LightGBM · imbalanced-learn · Optuna · MLflow · SHAP · FastAPI · Streamlit · Evidently AI · Docker

---

## Author

Built as a portfolio project targeting ML engineering and data science roles.
