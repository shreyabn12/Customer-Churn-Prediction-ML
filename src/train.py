# src/train.py
"""
train.py
--------
Full model training pipeline:
1. Load and preprocess data
2. Split into train/test sets
3. Apply SMOTE to training data only
4. Train 4 models with Optuna hyperparameter tuning
5. Track all experiments with MLflow
6. Save the best model to models/
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report,
    f1_score, precision_score, recall_score, confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Suppress noisy warnings during Optuna trials
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Add project root to path so we can import src modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_raw_data
from src.features import prepare_X_y, build_preprocessor, NUMERIC_COLS, MULTI_CAT_COLS

# ── Constants ────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
RANDOM_STATE = 42       # fixed seed so results are reproducible
TEST_SIZE = 0.2         # 80% train, 20% test
N_TRIALS = 30           # number of Optuna trials per model
CV_FOLDS = 5            # cross-validation folds


# ── Data preparation ─────────────────────────────────────────────────────────

def load_and_prepare_data():
    """
    Loads raw data, engineers features, encodes columns,
    and splits into train/test sets.
    
    Returns:
        X_train, X_test, y_train, y_test — all as DataFrames/Series
    """
    print("\n" + "="*60)
    print("STEP 1: Loading and preparing data")
    print("="*60)
    
    df = load_raw_data()
    X, y = prepare_X_y(df)
    
    # StratifiedKFold preserves class ratio in both train and test splits
    # Without stratify=y, by random chance test set might have very few churners
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y   # ensures 26.5% churners in both train AND test
    )
    
    print(f"\nTrain set: {X_train.shape[0]} rows")
    print(f"Test set:  {X_test.shape[0]} rows")
    print(f"Train churn rate: {y_train.mean():.3f}")
    print(f"Test churn rate:  {y_test.mean():.3f}")
    
    return X_train, X_test, y_train, y_test


# ── Model building helpers ───────────────────────────────────────────────────

def build_pipeline(classifier) -> ImbPipeline:
    """
    Wraps preprocessor + SMOTE + classifier into a single pipeline.
    
    Order matters critically:
    1. preprocessor: scales numerics, one-hot encodes categoricals
    2. SMOTE: creates synthetic minority samples (only on transformed data)
    3. classifier: learns from balanced, preprocessed data
    
    Using ImbPipeline from imbalanced-learn (not sklearn Pipeline)
    because sklearn's Pipeline doesn't support SMOTE as a step.
    
    CRITICAL: SMOTE is inside the pipeline, so it only runs during .fit()
    It never touches the test data. This prevents data leakage.
    """
    preprocessor = build_preprocessor()
    
    return ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ('classifier', classifier)
    ])


def evaluate_pipeline(pipeline, X_test, y_test) -> dict:
    """
    Evaluates a trained pipeline on the test set.
    Returns a dictionary of all metrics.
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]  # probability of churn
    
    return {
        'roc_auc':   round(roc_auc_score(y_test, y_prob), 4),
        'f1':        round(f1_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred), 4),
        'recall':    round(recall_score(y_test, y_pred), 4),
    }


# ── Optuna objective functions ───────────────────────────────────────────────
# Each function defines what hyperparameters to search and returns a CV score.
# Optuna calls these functions N_TRIALS times, each time trying different values.

def objective_lr(trial, X_train, y_train):
    """Logistic Regression hyperparameter search space."""
    params = {
        'C': trial.suggest_float('C', 0.001, 10.0, log=True),
        # C = inverse of regularization strength
        # Small C = strong regularization (simpler model, less overfit)
        # Large C = weak regularization (complex model, might overfit)
        
        'solver': trial.suggest_categorical('solver', ['lbfgs', 'saga']),
        'max_iter': 1000,
        'random_state': RANDOM_STATE
    }
    
    clf = LogisticRegression(**params)
    pipeline = build_pipeline(clf)
    
    # StratifiedKFold: splits training data into 5 folds, each with same churn ratio
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    return scores.mean()


def objective_rf(trial, X_train, y_train):
    """Random Forest hyperparameter search space."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        # number of decision trees in the forest
        
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        # how deep each tree can grow — deeper = more complex
        
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        # minimum samples required to split a node
        
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        # minimum samples required at a leaf node
        
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        # number of features to consider at each split
        
        'random_state': RANDOM_STATE,
        'n_jobs': -1  # use all CPU cores
    }
    
    clf = RandomForestClassifier(**params)
    pipeline = build_pipeline(clf)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=1)
    return scores.mean()


def objective_xgb(trial, X_train, y_train):
    """XGBoost hyperparameter search space."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        # how much each tree contributes — smaller = slower but more accurate
        
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        # fraction of training data used per tree — adds randomness, prevents overfit
        
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        # fraction of features used per tree
        
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        # L1 regularization
        
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        # L2 regularization
        
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    clf = XGBClassifier(**params)
    pipeline = build_pipeline(clf)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=1)
    return scores.mean()


def objective_lgbm(trial, X_train, y_train):
    """LightGBM hyperparameter search space."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        # LightGBM grows leaf-wise (not depth-wise like XGBoost)
        # num_leaves controls model complexity directly
        
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1  # suppress LightGBM output
    }
    
    clf = LGBMClassifier(**params)
    pipeline = build_pipeline(clf)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=1)
    return scores.mean()


# ── Main training function ───────────────────────────────────────────────────

def train_all_models(X_train, X_test, y_train, y_test):
    """
    Trains all 4 models with Optuna tuning and MLflow tracking.
    Returns a dict of results and saves the best model to disk.
    """
    
    # Configure MLflow — all runs go into mlruns/ folder in project root
    mlflow.set_experiment("churn-prediction")
    
    # Define which models to train and their objective functions
    model_configs = [
        ("LogisticRegression", objective_lr),
        ("RandomForest",       objective_rf),
        ("XGBoost",            objective_xgb),
        ("LightGBM",           objective_lgbm),
    ]
    
    # Map model names to their constructors for final training
    model_constructors = {
        "LogisticRegression": LogisticRegression,
        "RandomForest":       RandomForestClassifier,
        "XGBoost":            XGBClassifier,
        "LightGBM":           LGBMClassifier,
    }
    
    all_results = {}
    best_auc = 0
    best_model_name = None
    best_pipeline = None
    
    for model_name, objective_fn in model_configs:
        print(f"\n{'='*60}")
        print(f"TRAINING: {model_name}")
        print(f"Running {N_TRIALS} Optuna trials...")
        print(f"{'='*60}")
        
        # ── Optuna hyperparameter search ──
        study = optuna.create_study(
            direction='maximize',    # we want to MAXIMIZE AUC-ROC
            sampler=TPESampler(seed=RANDOM_STATE)
            # TPE = Tree-structured Parzen Estimator
            # It learns from previous trials to suggest better parameters
        )
        
        # Pass training data into the objective function
        study.optimize(
            lambda trial: objective_fn(trial, X_train, y_train),
            n_trials=N_TRIALS,
            show_progress_bar=False
        )
        
        best_params = study.best_params
        best_cv_auc = study.best_value
        print(f"Best CV AUC: {best_cv_auc:.4f}")
        print(f"Best params: {best_params}")
        
        # ── Train final model with best params on full training set ──
        # Add fixed params that aren't part of the search space
        if model_name == "LogisticRegression":
            best_params.update({'max_iter': 1000, 'random_state': RANDOM_STATE})
        elif model_name == "RandomForest":
            best_params.update({'random_state': RANDOM_STATE, 'n_jobs': -1})
        elif model_name == "XGBoost":
            best_params.update({
                'use_label_encoder': False,
                'eval_metric': 'logloss',
                'random_state': RANDOM_STATE,
                'n_jobs': -1
            })
        elif model_name == "LightGBM":
            best_params.update({
                'random_state': RANDOM_STATE,
                'n_jobs': -1,
                'verbose': -1
            })
        
        constructor = model_constructors[model_name]
        final_clf = constructor(**best_params)
        final_pipeline = build_pipeline(final_clf)
        final_pipeline.fit(X_train, y_train)
        
        # ── Evaluate on test set ──
        metrics = evaluate_pipeline(final_pipeline, X_test, y_test)
        all_results[model_name] = metrics
        
        print(f"\nTest set metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
        
        # ── Log everything to MLflow ──
        with mlflow.start_run(run_name=model_name):
            # Log hyperparameters
            mlflow.log_params(best_params)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("n_trials", N_TRIALS)
            mlflow.log_param("smote", True)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            mlflow.log_metric("cv_auc", best_cv_auc)
            
            # Log the pipeline (preprocessor + SMOTE + model) as one artifact
            mlflow.sklearn.log_model(final_pipeline, "pipeline")
        
        # Track the best model across all 4
        if metrics['roc_auc'] > best_auc:
            best_auc = metrics['roc_auc']
            best_model_name = model_name
            best_pipeline = final_pipeline
    
    # ── Save best model to disk ──
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    joblib.dump(best_pipeline, model_path)
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model_name} (AUC: {best_auc:.4f})")
    print(f"Saved to: {model_path}")
    print(f"{'='*60}")
    
    return all_results, best_model_name, best_pipeline


# ── Results summary ──────────────────────────────────────────────────────────

def print_results_summary(all_results: dict):
    """Prints a formatted comparison table of all models."""
    print("\n" + "="*60)
    print("FINAL MODEL COMPARISON")
    print("="*60)
    
    results_df = pd.DataFrame(all_results).T
    results_df = results_df.sort_values('roc_auc', ascending=False)
    print(results_df.to_string())
    print("\n(Higher is better for all metrics)")
    return results_df


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run the full training pipeline
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    all_results, best_name, best_pipeline = train_all_models(
        X_train, X_test, y_train, y_test
    )
    results_df = print_results_summary(all_results)
    
    # Save results table
    results_df.to_csv(
        os.path.join(os.path.dirname(__file__), '..', 'reports', 'model_comparison.csv')
    )
    print("\nResults saved to reports/model_comparison.csv")