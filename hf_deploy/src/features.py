# src/features.py
"""
features.py
-----------
Responsible for:
1. Feature engineering  — creating new columns from existing ones
2. Defining column types — which columns are numeric vs categorical
3. Building the sklearn ColumnTransformer — applies different transformations
   to different column types simultaneously

This module is imported by train.py in Phase 3.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# ── Column definitions ──────────────────────────────────────────────────────
# These lists tell our pipeline which columns to treat as numeric vs categorical
# We define them here once — imported everywhere else in the project

NUMERIC_COLS = [
    'tenure_months',
    'monthly_charges',
    'total_charges',
    'charges_per_month',      # engineered feature (created below)
    'tenure_group_encoded'    # engineered feature (created below)
]

# Binary columns: only 2 unique values (Yes/No, Male/Female, 0/1)
# We encode these as simple 0 and 1
BINARY_COLS = [
    'gender',
    'senior_citizen',
    'partner',
    'dependents',
    'phone_service',
    'paperless_billing'
]

# Multi-category columns: 3 or more unique values
# These get One-Hot Encoded (explained below)
MULTI_CAT_COLS = [
    'multiple_lines',
    'internet_service',
    'online_security',
    'online_backup',
    'device_protection',
    'tech_support',
    'streaming_tv',
    'streaming_movies',
    'contract',
    'payment_method'
]


# ── Feature engineering ──────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features from existing columns.
    Always call this BEFORE building the preprocessing pipeline.
    
    New features created:
    1. charges_per_month  : average monthly spend = total_charges / tenure_months
                            Customers who spend more per month may churn more
    2. tenure_group       : groups tenure into buckets (new/mid/long-term)
                            Raw tenure is noisy; groups can be more predictive
    3. tenure_group_encoded: numeric version of tenure_group for the model
    
    Parameters:
        df: DataFrame with cleaned columns from data_loader.py
    
    Returns:
        df: DataFrame with new feature columns added
    """
    df = df.copy()  # never modify the original dataframe
    
    # Feature 1: charges_per_month
    # For new customers (tenure=0), avoid division by zero by using monthly_charges directly
    df['charges_per_month'] = np.where(
        df['tenure_months'] == 0,
        df['monthly_charges'],
        df['total_charges'] / df['tenure_months']
    )
    df['charges_per_month'] = df['charges_per_month'].round(2)
    
    # Feature 2: tenure_group — bucket raw tenure into 4 meaningful groups
    # 0-12 months   = new customers (highest churn risk)
    # 13-24 months  = developing loyalty
    # 25-48 months  = established customers
    # 49+ months    = long-term loyal customers
    df['tenure_group'] = pd.cut(
        df['tenure_months'],
        bins=[0, 12, 24, 48, 72],
        labels=['new', 'developing', 'established', 'loyal'],
        include_lowest=True
    )
    
    # Feature 3: numeric version of tenure_group (0,1,2,3)
    # pd.cut creates a Categorical type — models need numbers
    tenure_map = {'new': 0, 'developing': 1, 'established': 2, 'loyal': 3}
    df['tenure_group_encoded'] = df['tenure_group'].map(tenure_map)
    
    print(f"[features] Engineered features added: charges_per_month, tenure_group, tenure_group_encoded")
    print(f"[features] New shape: {df.shape}")
    return df


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts binary text columns to 0 and 1.
    
    Examples:
        'Yes' -> 1,  'No' -> 0
        'Male' -> 1, 'Female' -> 0
        1 -> 1,      0 -> 0  (senior_citizen is already numeric in some versions)
    
    Parameters:
        df: DataFrame after engineer_features()
    
    Returns:
        df: DataFrame with binary columns replaced by 0/1 integers
    """
    df = df.copy()
    
    for col in BINARY_COLS:
        if col not in df.columns:
            continue
            
        # If already numeric (0/1), leave it alone
        if df[col].dtype in ['int64', 'float64']:
            continue
        
        unique_vals = df[col].dropna().unique()
        
        # Yes/No columns
        if set(unique_vals) <= {'Yes', 'No'}:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
        
        # Male/Female columns
        elif set(unique_vals) <= {'Male', 'Female'}:
            df[col] = df[col].map({'Male': 1, 'Female': 0})
        
        else:
            # Fallback: use LabelEncoder for anything else
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    print(f"[features] Binary columns encoded: {BINARY_COLS}")
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Builds a sklearn ColumnTransformer that applies:
    
    1. StandardScaler to numeric columns
       - Scales values so mean=0 and std=1
       - Example: tenure_months [0-72] becomes roughly [-1.5 to 1.5]
       - Why: prevents the model treating "72 months" as 72x more important than "1 month"
    
    2. OneHotEncoder to multi-category columns
       - Converts 'Month-to-month', 'One year', 'Two year' into 3 separate 0/1 columns
       - Why: there's no natural ordering between contract types, so we can't just use 0,1,2
       - handle_unknown='ignore': if new data has an unexpected category, don't crash
    
    The ColumnTransformer applies these transformations in parallel and combines the results.
    
    Returns:
        preprocessor: sklearn ColumnTransformer object
    """
    
    # Pipeline for numeric columns: just scale them
    numeric_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
        # StandardScaler formula: (value - mean) / std_deviation
    ])
    
    # Pipeline for multi-category columns: one-hot encode them
    categorical_pipeline = Pipeline(steps=[
        ('onehot', OneHotEncoder(
            handle_unknown='ignore',  # don't crash on unseen categories
            sparse_output=False       # return a regular array, not a sparse matrix
        ))
    ])
    
    # ColumnTransformer applies each pipeline to its designated columns
    # remainder='passthrough' means: keep any columns not listed above as-is
    # (our binary cols are already 0/1 so they just pass through)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, NUMERIC_COLS),
            ('cat', categorical_pipeline, MULTI_CAT_COLS)
        ],
        remainder='passthrough'  # binary cols pass through unchanged
    )
    
    print("[features] Preprocessor (ColumnTransformer) built successfully")
    return preprocessor


def prepare_X_y(df: pd.DataFrame):
    """
    Full preprocessing pipeline:
    1. Engineer new features
    2. Encode binary columns
    3. Separate X and y
    4. Encode the target y (Yes->1, No->0)
    
    This is the main function called by train.py
    
    Returns:
        X (DataFrame): fully preprocessed features, ready for the ColumnTransformer
        y (Series): target as 0/1 integers
    """
    from src.data_loader import get_feature_target_split
    
    # Step 1: engineer features
    df = engineer_features(df)
    
    # Step 2: encode binary columns  
    df = encode_binary_columns(df)
    
    # Step 3: split into X and y
    # We need to re-split because engineer_features added new columns
    # Drop the text version of tenure_group (we keep the encoded version)
    cols_to_drop = [
        'customerid', 'count', 'country', 'state',
        'lat_long', 'city', 'zip_code', 'latitude', 'longitude',
        'churn_value', 'churn_score', 'churn_reason', 'cltv',
        'tenure_group',  # text version — we use tenure_group_encoded instead
        'churn_label'
    ]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    X = df.drop(columns=cols_to_drop)
    y = df['churn_label'].map({'Yes': 1, 'No': 0})
    
    print(f"\n[features] Final X shape: {X.shape}")
    print(f"[features] Final y shape: {y.shape}")
    print(f"[features] y distribution: {y.value_counts().to_dict()}")
    print(f"[features] X columns: {X.columns.tolist()}")
    
    return X, y