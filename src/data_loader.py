# src/data_loader.py
"""
data_loader.py
--------------
Single responsibility: load the raw Excel file and return a clean DataFrame.

Key things this file does differently from a basic project:
1. Reads an .xlsx file (not CSV) using openpyxl engine
2. Cleans column names automatically (removes spaces, lowercases everything)
3. Raises a helpful error if the file is missing
4. Can be imported by any other module in the project
"""

import pandas as pd
import os

# Build the path to the raw data file relative to this file's location
# os.path.dirname(__file__) = the src/ folder
# ".." goes up one level to the project root
# Then we go into data/raw/
RAW_DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "data", "raw", "Telco_customer_churn.xlsx"
)

def load_raw_data(path: str = 'C:/p/data/raw/Telco_customer_churn.xlsx') -> pd.DataFrame:
    """
    Loads the raw IBM Telco churn Excel file into a pandas DataFrame.
    Cleans column names: strips spaces, lowercases, replaces spaces with underscores.
    
    Example: 'Churn Label' becomes 'churn_label', 'Tenure Months' becomes 'tenure_months'
    
    Returns:
        pd.DataFrame with 33 columns and ~7,000 rows
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at: {path}\n"
            "Please download from Kaggle and place in data/raw/ folder.\n"
            "Rename it to: Telco_customer_churn.xlsx"
        )
    
    # engine='openpyxl' is required for .xlsx files
    df = pd.read_excel('C:/p/data/raw/Telco_customer_churn.xlsx', engine='openpyxl')
    
    # Clean column names so we never have to deal with spaces again
    # 'Churn Label' -> 'churn_label'
    # 'Zip Code'    -> 'zip_code'
    # 'Total Charges' -> 'total_charges'
    df.columns = (
        df.columns
        .str.strip()           # remove leading/trailing spaces
        .str.lower()           # make lowercase
        .str.replace(' ', '_') # replace spaces with underscores
        .str.replace('-', '_') # replace hyphens with underscores
    )
    
    print(f"[data_loader] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[data_loader] Columns: {df.columns.tolist()}")
    return df


def get_feature_target_split(df: pd.DataFrame):
    """
    Separates the dataset into features (X) and target (y).
    
    We use 'churn_label' as target (Yes/No).
    We drop columns that would leak information or are not useful for ML:
    
    - customer_id     : unique identifier, not a pattern
    - count           : always 1, useless
    - country, state  : everyone is from same region in this dataset
    - lat_long        : raw string combining latitude and longitude
    - churn_value     : this IS the target encoded as 0/1 — using it would be cheating
    - churn_score     : a score already computed from churn, data leakage
    - churn_reason    : only available AFTER someone churned, data leakage
    - cltv            : customer lifetime value, can cause leakage
    
    Returns:
        X (DataFrame): features only
        y (Series): target column (churn_label)
    """
    # Columns to drop — these either leak the answer or have no predictive value
    cols_to_drop = [
        'customer_id', 'count', 'country', 'state',
        'lat_long', 'churn_value', 'churn_score', 'churn_reason', 'cltv'
    ]
    
    # Only drop columns that actually exist (defensive coding)
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    X = df.drop(columns=cols_to_drop + ['churn_label'])
    y = df['churn_label']  # 'Yes' or 'No'
    
    print(f"[data_loader] Features shape: {X.shape}")
    print(f"[data_loader] Target shape: {y.shape}")
    print(f"[data_loader] Feature columns: {X.columns.tolist()}")
    return X, y


if __name__ == "__main__":
    # Quick test — run this file directly with: python src/data_loader.py
    df = load_raw_data()
    print("\nFirst 3 rows:")
    print(df.head(3))
    print("\nColumn dtypes:")
    print(df.dtypes)
    X, y = get_feature_target_split(df)
    print("\nTarget distribution:")
    print(y.value_counts())