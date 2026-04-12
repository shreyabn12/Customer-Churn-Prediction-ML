# Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Best%20Model-orange?logo=xgboost)
![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.8535-brightgreen)
![MLflow](https://img.shields.io/badge/MLflow-Tracked-blue?logo=mlflow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

An end-to-end, production-grade machine learning system that predicts customer churn using the IBM Telco Customer Churn dataset. Built with a focus on clean modular code, rigorous experiment tracking, model explainability, and real-world deployment practices — targeting the standards expected in ML engineering roles at top tech companies.

---

## 🔗 Live Demo

| Service | Link |
|---|---|
| REST API (FastAPI) | *Coming after Phase 7 — Deployment* |
| Interactive Dashboard (Streamlit) | *Coming after Phase 7 — Deployment* |
| MLflow Experiment Tracker | Run `mlflow ui` locally → http://127.0.0.1:5000 |

---

## 📌 Project Overview

Customer churn — when a customer stops using a service — is one of the most costly problems in subscription businesses. Acquiring a new customer costs five to seven times more than retaining an existing one, which means even a small improvement in churn prediction can save millions in revenue.

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
          │  Docker Container│
          │  (Deployed on    │
          │  Render / HF)    │
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
│   └── 03_explain.ipynb           # SHAP explainability visualizations
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Loads .xlsx, cleans columns, fixes dtypes
│   ├── features.py                # Feature engineering + ColumnTransformer
│   ├── train.py                   # Full training pipeline (4 models + MLflow)
│   ├── evaluate.py                # Model evaluation utilities
│   └── predict.py                 # Prediction helper used by the API
│
├── api/
│   ├── __init__.py
│   └── main.py                    # FastAPI REST endpoint (/predict)
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
├── config/                        # Configuration files
├── Dockerfile                     # Container definition for deployment
├── requirements.txt               # All Python dependencies with pinned versions
└── README.md
```

---

## 📊 Dataset

**IBM Telco Customer Churn** — sourced from [Kaggle](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset/data)

The dataset contains **7,043 customer records** with **33 columns** covering demographics, account information, and subscribed services. The target variable is `Churn Label` (Yes/No), with a natural class imbalance of approximately **73.5% No / 26.5% Yes**.

The following columns were identified as **data leakage** and excluded from all model training, because they are derived from or recorded *after* the churn event — information that would not be available at real prediction time:

- `Churn Value` — the target encoded as 0/1 (using this would be literally giving the model the answer)
- `Churn Score` — an IBM-computed score that already incorporates churn probability
- `Churn Reason` — only recorded after a customer has already churned
- `CLTV` — Customer Lifetime Value incorporates churn probability in its formula

Geographic columns (`City`, `Zip Code`, `Latitude`, `Longitude`, `Lat Long`) were also dropped as they are too granular to generalize and the dataset does not have enough geographic diversity to learn from.

---

## ⚙️ Feature Engineering

Two new features were created from the raw columns to improve predictive signal.

**`charges_per_month`** is computed as `total_charges / tenure_months` and represents a customer's average monthly spend normalized by how long they've been a customer. New customers with zero tenure use `monthly_charges` directly to avoid division by zero. This feature captures spending consistency better than either raw column alone — a customer paying more per month than average for their tenure group is often at higher risk.

**`tenure_group_encoded`** buckets raw tenure in months into four ordinal groups: new (0–12 months, highest risk), developing (13–24), established (25–48), and loyal (49+). The raw tenure number is noisy — the difference in churn risk between month 1 and month 12 is far greater than between month 50 and month 60. Bucketing captures this non-linear relationship more cleanly.

---

## 🤖 Models Trained

All four models were trained using a consistent end-to-end pipeline:

```
ColumnTransformer  →  SMOTE  →  Classifier
```

Hyperparameters were tuned using **Optuna** (30 trials per model, TPE sampler) with **5-fold stratified cross-validation** scoring on AUC-ROC. All runs are logged and reproducible in MLflow.

| Model | AUC-ROC | F1 Score | Precision | Recall |
|---|---|---|---|---|
| **XGBoost ✅ (Best)** | **0.8535** | **0.6355** | 0.5761 | 0.7086 |
| LightGBM | 0.8535 | 0.6333 | **0.5833** | 0.6925 |
| Random Forest | 0.8510 | 0.6309 | 0.5509 | 0.7380 |
| Logistic Regression | 0.8462 | 0.6149 | 0.5106 | **0.7727** |

XGBoost was selected as the best model based on AUC-ROC score. Notably, **Logistic Regression achieves the highest recall (0.7727)**, meaning it catches the most actual churners — which may be preferable in a business context where missing a churner (false negative) is more costly than a false alarm (false positive). The right model to deploy ultimately depends on the cost structure the business assigns to each type of error.

---

## 🔑 Key Technical Decisions

**Why SMOTE inside the Pipeline?** SMOTE is placed as a step inside an `imblearn.Pipeline` rather than applied directly to the full training dataset before cross-validation. This is critical: if SMOTE runs before cross-validation, synthetic samples derived from validation-fold rows leak into the training fold, inflating AUC estimates. Inside the pipeline, SMOTE only ever sees training fold data during each cross-validation split, giving honest performance estimates.

**Why AUC-ROC over Accuracy?** With a 73.5/26.5 class split, a model that always predicts "No Churn" achieves 73.5% accuracy while being completely useless. AUC-ROC is threshold-independent and measures how well the model separates the two classes across all possible decision thresholds — it is the industry standard metric for binary classification on imbalanced data.

**Why four models instead of one?** No single algorithm is always best. Logistic Regression provides a strong interpretable baseline, Random Forest captures non-linear patterns through ensemble averaging, and XGBoost / LightGBM represent the state of the art for tabular data. Comparing all four surfaces the precision-recall tradeoff clearly and gives the business a choice of operating point.

**Why `ColumnTransformer`?** Different column types need different transformations — numeric columns need `StandardScaler`, multi-category columns need `OneHotEncoder`, and binary columns just need a 0/1 mapping. `ColumnTransformer` applies all of these in parallel in a single step, keeping the code clean and ensuring the test set is always transformed using statistics learned only from the training set (preventing leakage).

---

## 🔍 SHAP Explainability

Model predictions are explained using **SHAP (SHapley Additive exPlanations)**, which fairly distributes each feature's contribution to any individual prediction using cooperative game theory.

Three visualization types are produced and saved to `reports/figures/`:

The **summary plot** (`shap_summary.png`) shows global feature importance across all test customers — which features move predictions the most, in which direction, and with how much variance across the population. Red means a high feature value increases churn risk; blue means it decreases it.

The **waterfall plot** (`shap_waterfall_high_risk.png`) explains a single high-risk customer's prediction in full detail, showing exactly which factors drove their predicted churn probability and by how much — this is what a customer success manager would use before making a retention call.

The **dependence plot** (`shap_dependence_tenure.png`) shows how churn risk varies continuously with tenure, revealing that the relationship is strongly non-linear and most pronounced in the first 12 months — after which the marginal churn risk from one additional month of tenure flattens considerably.

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

# Mac / Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate.bat

pip install -r requirements.txt
```

**3. Download the dataset:**

Download `Telco_customer_churn.xlsx` from [this Kaggle page](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset/data), rename it to exactly `Telco_customer_churn.xlsx`, and place it in `data/raw/`.

**4. Run exploratory data analysis:**

```bash
jupyter notebook notebooks/01_eda.ipynb
```

**5. Train all four models:**

```bash
python src/train.py
```

This will run 30 Optuna trials per model (approximately 5–10 minutes total) and save the best model pipeline to `models/best_model.pkl`.

**6. View experiment results in MLflow:**

```bash
mlflow ui
# Then open http://127.0.0.1:5000 in your browser
```

**7. Run the SHAP explainability notebook:**

```bash
jupyter notebook notebooks/03_explain.ipynb
```

**8. Start the FastAPI prediction server** *(available after Phase 5)*:

```bash
uvicorn api.main:app --reload
# Docs available at http://127.0.0.1:8000/docs
```

**9. Launch the Streamlit dashboard** *(available after Phase 6)*:

```bash
streamlit run dashboard/app.py
```

---

## 🔌 API Usage

Once the FastAPI server is running, send a POST request to `/predict` with a customer's details in JSON format:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 2,
    "monthly_charges": 70.5,
    "total_charges": 141.0,
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

The response will include the predicted churn probability and the binary prediction:

```json
{
  "churn_probability": 0.847,
  "churn_prediction": "Yes",
  "risk_level": "High"
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
| Phase 5 | FastAPI REST endpoint | 🔲 In Progress |
| Phase 6 | Streamlit interactive dashboard | 🔲 Pending |
| Phase 7 | Docker containerization | 🔲 Pending |
| Phase 8 | Deployment on Render / Hugging Face Spaces | 🔲 Pending |
| Phase 9 | Evidently AI data drift monitoring | 🔲 Pending |

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
| Containerization | Docker |
| Deployment | Render / Hugging Face Spaces |

---

## 👤 Author

**Shreya B N**
Built as a portfolio project targeting ML engineering and data science roles at top technology companies.



---

