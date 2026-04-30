# Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Best%20Model-orange)
![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.8535-brightgreen)
![MLflow](https://img.shields.io/badge/MLflow-Tracked-blue?logo=mlflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

An end-to-end, production-grade machine learning system that predicts customer churn using the IBM Telco Customer Churn dataset. Built with a focus on clean modular code, rigorous experiment tracking, model explainability, and real-world deployment — targeting the standards expected in ML engineering roles at top technology companies.

---

## 🔗 Live Demo

| Service | Link |
|---|---|
| REST API (FastAPI + Swagger UI) | https://customer-churn-prediction-ml-xart.onrender.com/ |
| Interactive Dashboard (Streamlit) | https://shreyabn-churn-prediction-dashboard.hf.space |
| MLflow Experiment Tracker | Run `mlflow ui` locally → http://127.0.0.1:5000 |

> **Note:** The Render API may take 30–60 seconds to wake up on the first request due to free-tier sleep mode. This is expected behaviour.

---

## 📌 Project Overview

Customer churn — when a customer stops using a service — is one of the most costly problems in subscription businesses. Acquiring a new customer costs five to seven times more than retaining an existing one, meaning even a small improvement in churn prediction can save millions in revenue.

This project builds a complete ML pipeline that not only predicts which customers are at risk of churning, but also explains **why** the model makes each individual prediction — making it directly actionable for business and customer success teams.

The system achieves an **AUC-ROC of 0.8535** on the held-out test set using XGBoost with Optuna-driven hyperparameter tuning and SMOTE for class imbalance correction. All four trained models, their parameters, and their metrics are tracked and reproducible via MLflow.

---

## 🏗️ System Architecture

```
Raw Excel Data (Kaggle)
        │
        ▼
┌─────────────────────────────┐
│   src/data_loader.py        │  Loads .xlsx, cleans column names,
│                             │  fixes data types, drops leakage cols
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   src/features.py           │  Engineers new features, encodes
│                             │  binary cols, builds ColumnTransformer
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│   sklearn + imblearn Pipeline                       │
│                                                     │
│   ColumnTransformer  →  SMOTE  →  Classifier        │
│   (scale + encode)     (balance)   (LR/RF/XGB/LGBM) │
└────────────┬────────────────────────────────────────┘
             │
     ┌───────┴────────┐
     ▼                ▼
┌─────────┐     ┌───────────┐
│ MLflow  │     │  models/  │
│Tracking │     │best_model │
│  UI     │     │  .pkl     │
└─────────┘     └─────┬─────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌──────────────────┐   ┌─────────────────────┐
│  FastAPI REST    │   │  Streamlit Dashboard │
│  /predict        │   │  Interactive UI +    │
│  endpoint        │   │  SHAP explanations   │
└────────┬─────────┘   └──────────┬──────────┘
         │                        │
         └──────────┬─────────────┘
                    ▼
          ┌──────────────────┐
          │  Deployed on     │
          │  Render +        │
          │  Hugging Face    │
          └──────────────────┘
                    │
                    ▼
          ┌──────────────────┐
          │  Evidently AI    │
          │  Data Drift      │
          │  Monitoring      │
          └──────────────────┘
```

---

## 📂 Project Structure

```
churn-prediction/
│
├── data/
│   ├── raw/                        # Original dataset from Kaggle — never modified
│   └── processed/                  # Cleaned and engineered data saved here
│
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory data analysis (14 cells)
│   ├── 02_train.ipynb             # Model training + Optuna tuning
│   ├── 03_explain.ipynb           # SHAP explainability visualizations
│   └── 04_api_test.ipynb          # API testing notebook
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Loads .xlsx, cleans columns, fixes dtypes
│   ├── features.py                # Feature engineering + ColumnTransformer
│   ├── train.py                   # Full training pipeline (4 models + MLflow)
│   ├── evaluate.py                # Model evaluation utilities
│   ├── predict.py                 # Prediction helper used by the API
│   └── monitor_drift.py           # Evidently AI drift monitoring
│
├── api/
│   ├── __init__.py
│   └── main.py                    # FastAPI REST endpoint (/predict, /predict/batch)
│
├── dashboard/
│   └── app.py                     # Streamlit interactive dashboard
│
├── models/
│   └── best_model.pkl             # Saved best pipeline (preprocessor + SMOTE + XGBoost)
│
├── reports/
│   ├── figures/                   # All saved plots (SHAP, distributions, confusion matrix)
│   ├── drift/                     # Evidently AI HTML drift reports
│   └── model_comparison.csv       # Final metrics table across all 4 models
│
├── render.yaml                    # Render deployment configuration
├── requirements.txt               # All Python dependencies
└── README.md
```

---

## 📊 Dataset

**IBM Telco Customer Churn** — sourced from [Kaggle](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset/data)

The dataset contains **7,043 customer records** with **33 columns** covering demographics, account information, and subscribed services. The target variable is `Churn Label` (Yes/No), with a natural class imbalance of approximately **73.5% No / 26.5% Yes**.

The following columns were identified as **data leakage** and excluded from all model training:

- `Churn Value` — the target encoded as 0/1 (giving the model the answer directly)
- `Churn Score` — an IBM-computed score that already incorporates churn probability
- `Churn Reason` — only recorded after a customer has already churned
- `CLTV` — Customer Lifetime Value incorporates churn probability in its formula

Geographic columns (`City`, `Zip Code`, `Latitude`, `Longitude`) were also dropped as they are too granular to generalize.

---

## ⚙️ Feature Engineering

Two new features were created from raw columns to improve predictive signal.

**`charges_per_month`** is computed as `total_charges / tenure_months` and represents a customer's average monthly spend normalized by how long they have been a customer. New customers with zero tenure use `monthly_charges` directly to avoid division by zero.

**`tenure_group_encoded`** buckets raw tenure into four ordinal groups: new (0–12 months), developing (13–24), established (25–48), and loyal (49+). The raw tenure number is noisy — the difference in churn risk between month 1 and month 12 is far greater than between month 50 and month 60. Bucketing captures this non-linear relationship more cleanly.

---

## 🤖 Models Trained

All four models were trained using a consistent end-to-end pipeline:

```
ColumnTransformer  →  SMOTE  →  Classifier
```

Hyperparameters were tuned using **Optuna** (30 trials per model, TPE sampler) with **5-fold stratified cross-validation** scoring on AUC-ROC. All runs are logged and reproducible in MLflow.

| Model | AUC-ROC | F1 Score | Precision | Recall |
|---|---|---|---|---|
| **XGBoost (Best)** | **0.8535** | **0.6355** | 0.5761 | 0.7086 |
| LightGBM | 0.8535 | 0.6333 | **0.5833** | 0.6925 |
| Random Forest | 0.8510 | 0.6309 | 0.5509 | 0.7380 |
| Logistic Regression | 0.8462 | 0.6149 | 0.5106 | **0.7727** |

XGBoost was selected as the best model based on AUC-ROC. Notably, **Logistic Regression achieves the highest recall (0.7727)** — it catches the most actual churners, which may be preferable in contexts where missing a churner is more costly than a false alarm.

---

## 🔑 Key Technical Decisions

**Why SMOTE inside the Pipeline?** SMOTE is placed as a step inside an `imblearn.Pipeline` rather than applied directly to the full dataset before cross-validation. This is critical — if SMOTE runs before cross-validation, synthetic samples derived from validation-fold rows leak into training folds, inflating AUC estimates. Inside the pipeline, SMOTE only ever sees training fold data during each split.

**Why AUC-ROC over Accuracy?** With a 73.5/26.5 class split, a model that always predicts "No Churn" achieves 73.5% accuracy while being completely useless. AUC-ROC is threshold-independent and measures how well the model separates the two classes across all possible decision thresholds.

**Why four models instead of one?** No single algorithm is always best. Logistic Regression provides a strong interpretable baseline, Random Forest captures non-linear patterns through ensemble averaging, and XGBoost/LightGBM represent the state of the art for tabular data. Comparing all four surfaces the precision-recall tradeoff clearly.

**Why drop CLTV and Churn Score?** These are IBM-computed scores that incorporate churn probability. Including them would make the model appear extremely accurate but would be meaningless in production where those scores are not available at prediction time.

---

## 🔍 SHAP Explainability

Model predictions are explained using **SHAP (SHapley Additive exPlanations)**. Three visualization types are produced:

The **summary plot** shows global feature importance across all test customers — the top predictors were `contract_Month-to-month`, `dependents`, and `tenure_months`, confirming business intuition.

The **waterfall plot** explains a single high-risk customer's prediction in full detail, showing which specific factors drove their predicted churn probability.

The **dependence plot** shows how churn risk varies continuously with tenure, revealing that the relationship is strongly non-linear and most pronounced in the first 12 months.

---

## 📡 Data Drift Monitoring

Evidently AI is used to monitor whether incoming production data starts to drift from the training distribution. The monitoring script compares reference (training) data against current (new) data and generates an interactive HTML report.

```bash
python src/monitor_drift.py
# Report saved to: reports/drift/drift_report.html
```

---

## 🚀 How to Run Locally

**1. Clone the repository:**

```bash
git clone https://github.com/shreyabn12/Customer-Churn-Prediction-ML.git
cd Customer-Churn-Prediction-ML
```

**2. Create a virtual environment and install dependencies:**

```bash
python -m venv venv

# Windows:
venv\Scripts\activate.bat

# Mac / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

**3. Download the dataset:**

Download `Telco_customer_churn.xlsx` from [Kaggle](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset/data), rename it to `Telco_customer_churn.xlsx`, and place it in `data/raw/`.

**4. Train all models:**

```bash
python src/train.py
```

**5. View MLflow experiments:**

```bash
mlflow ui
# Open http://127.0.0.1:5000
```

**6. Start the FastAPI server:**

```bash
uvicorn api.main:app --reload
# Docs at http://127.0.0.1:8000/docs
```

**7. Launch the Streamlit dashboard:**

```bash
streamlit run dashboard/app.py
# Opens at http://localhost:8501
```

**8. Run drift monitoring:**

```bash
python src/monitor_drift.py
# Report saved to reports/drift/drift_report.html
```

---

## 🔌 API Usage

Send a POST request to `/predict`:

```bash
curl -X POST "https://churn-prediction-api.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 2,
    "monthly_charges": 85.5,
    "total_charges": 171.0,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check",
    "gender": "Male",
    "senior_citizen": 0,
    "partner": "No",
    "dependents": "No",
    "phone_service": "Yes",
    "multiple_lines": "No",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "paperless_billing": "Yes"
  }'
```

**Response:**

```json
{
  "churn_prediction": "Yes",
  "churn_probability": 0.8724,
  "risk_level": "High",
  "confidence": "Model is 87% confident this customer will churn.",
  "model_used": "XGBClassifier"
}
```

---

## ✅ Project Progress

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Project structure, virtual environment, data loading | ✅ Complete |
| Phase 2 | Feature engineering and preprocessing pipeline | ✅ Complete |
| Phase 3 | Model training (4 models), Optuna tuning, SMOTE, MLflow | ✅ Complete |
| Phase 4 | SHAP explainability (summary, waterfall, dependence plots) | ✅ Complete |
| Phase 5 | FastAPI REST endpoint with batch prediction | ✅ Complete |
| Phase 6 | Streamlit interactive dashboard | ✅ Complete |
| Phase 7 | Docker containerization | ⏭️ Skipped (Windows elevation restriction) |
| Phase 8 | Deployment — Render (API) + Hugging Face Spaces (Dashboard) | ✅ Complete |
| Phase 9 | Evidently AI data drift monitoring | ✅ Complete |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn, XGBoost, LightGBM |
| Class Imbalance | imbalanced-learn (SMOTE) |
| Hyperparameter Tuning | Optuna (TPE Sampler) |
| Experiment Tracking | MLflow |
| Explainability | SHAP |
| API | FastAPI, Uvicorn |
| Dashboard | Streamlit |
| Drift Monitoring | Evidently AI |
| Deployment | Render (API), Hugging Face Spaces (Dashboard) |

---

## 👤 Author

**Shreya B N**
Built as a portfolio project targeting ML engineering and data science roles at top technology companies.

[![GitHub](https://img.shields.io/badge/GitHub-shreyabn12-black?logo=github)](https://github.com/shreyabn12)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-ShreyaBN-yellow?logo=huggingface)](https://huggingface.co/ShreyaBN)
