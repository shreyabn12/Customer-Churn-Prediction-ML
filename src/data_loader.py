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
   # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('-', '_')
    )

    # Fix total_charges — Excel stores blank cells as empty strings
    # pd.to_numeric with errors='coerce' converts blanks to NaN
    # fillna(0) replaces NaN with 0 (brand new customers with no charges yet)
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce').fillna(0)

    print(f"[data_loader] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[data_loader] Columns: {df.columns.tolist()}")
    return df


def get_feature_target_split(df: pd.DataFrame):
    """
    Separates the dataset into features (X) and target (y).
    Drops columns that cause data leakage or have no ML value.
    """
    cols_to_drop = [
        'customerid',    # unique ID — no pattern to learn
        'count',         # always 1, useless
        'country',       # everyone is USA
        'state',         # everyone is California
        'city',          # too many unique values, not useful
        'zip_code',      # too granular
        'latitude',      # raw geo coordinate
        'longitude',     # raw geo coordinate
        'lat_long',      # string combining lat+long
        'churn_value',   # 0/1 encoding of our target — data leakage!
        'churn_score',   # computed FROM churn — data leakage!
        'churn_reason',  # only known AFTER churn — data leakage!
        'cltv',          # customer lifetime value — leakage risk
    ]

    # Only drop columns that actually exist in the dataframe
    # This prevents errors if some columns are missing
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