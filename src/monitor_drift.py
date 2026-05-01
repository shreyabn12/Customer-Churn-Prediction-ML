# src/monitor_drift.py
"""
Data Drift Monitoring — Manual Statistical Implementation.

Compares training data distribution against new data to detect drift.
Uses scipy for statistical tests instead of Evidently (avoids version conflicts).

Run with: python src/monitor_drift.py
Output:   reports/drift/drift_report.html
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from scipy import stats

from src.data_loader import load_raw_data
from src.features import prepare_X_y

DRIFT_REPORT_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'reports', 'drift', 'drift_report.html'
)


def check_drift(reference: pd.Series, current: pd.Series, col: str) -> dict:
    """
    Uses the Kolmogorov-Smirnov test for numeric columns
    and chi-squared test for categorical columns.

    KS test: compares the full distribution shape of two samples.
    A p-value below 0.05 means the distributions are significantly different
    — i.e., drift has occurred.
    """
    if reference.dtype in ['int64', 'float64']:
        stat, p_value = stats.ks_2samp(
            reference.dropna().values,
            current.dropna().values
        )
        test_used = "KS Test"
    else:
        # For categorical: compare value frequencies
        ref_counts = reference.value_counts(normalize=True)
        cur_counts = current.value_counts(normalize=True)
        all_cats = set(ref_counts.index) | set(cur_counts.index)
        ref_freq = [ref_counts.get(c, 0) for c in all_cats]
        cur_freq = [cur_counts.get(c, 0) for c in all_cats]
        # Chi-squared needs counts not proportions
        n = len(current)
        expected = [f * n for f in ref_freq]
        observed = [f * n for f in cur_freq]
        # Avoid zeros
        expected = [max(e, 0.001) for e in expected]
        stat, p_value = stats.chisquare(observed, expected)
        test_used = "Chi-Squared"

    drifted = p_value < 0.05
    ref_mean = reference.mean() if reference.dtype in ['int64', 'float64'] else "N/A"
    cur_mean = current.mean() if current.dtype in ['int64', 'float64'] else "N/A"

    return {
        'column': col,
        'test': test_used,
        'statistic': round(float(stat), 4),
        'p_value': round(float(p_value), 4),
        'drifted': drifted,
        'ref_mean': round(float(ref_mean), 3) if ref_mean != "N/A" else "N/A",
        'cur_mean': round(float(cur_mean), 3) if cur_mean != "N/A" else "N/A",
    }


def run_drift_report():
    print("[monitor] Loading data...")
    df = load_raw_data()
    X, y = prepare_X_y(df)
    X['churn'] = y.values

    # Reference = first 70%, Current = last 30%
    split = int(len(X) * 0.7)
    reference = X.iloc[:split].copy()
    current = X.iloc[split:].copy()

    print(f"[monitor] Reference data: {len(reference)} rows")
    print(f"[monitor] Current data:   {len(current)} rows")
    print("[monitor] Running drift analysis...")

    # Columns to check
    cols_to_check = [
        'tenure_months', 'monthly_charges', 'total_charges',
        'charges_per_month', 'tenure_group_encoded',
        'gender', 'senior_citizen', 'partner', 'dependents',
        'contract', 'internet_service', 'payment_method'
    ]
    cols_to_check = [c for c in cols_to_check if c in reference.columns]

    results = []
    for col in cols_to_check:
        result = check_drift(reference[col], current[col], col)
        results.append(result)

    results_df = pd.DataFrame(results)
    drifted_cols = results_df[results_df['drifted'] == True]

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("DRIFT MONITORING REPORT")
    print("="*70)
    print(f"{'Column':<30} {'Test':<14} {'p-value':>8} {'Drift':>8}")
    print("-"*70)
    for _, row in results_df.iterrows():
        flag = "DRIFT" if row['drifted'] else "OK"
        print(f"{row['column']:<30} {row['test']:<14} {row['p_value']:>8.4f} {flag:>8}")
    print("="*70)
    print(f"\nTotal columns checked: {len(results_df)}")
    print(f"Drifted columns:       {len(drifted_cols)}")
    print(f"Stable columns:        {len(results_df) - len(drifted_cols)}")
    print()
    if len(drifted_cols) == 0:
        print("STATUS: No drift detected. Model inputs are stable.")
        print("ACTION: No retraining needed at this time.")
    elif len(drifted_cols) <= 2:
        print("STATUS: Minor drift detected in a few columns.")
        print("ACTION: Monitor closely. Retraining may be needed soon.")
    else:
        print("STATUS: Significant drift detected.")
        print("ACTION: Retrain the model with recent data.")

    # ── Save HTML report ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(DRIFT_REPORT_PATH), exist_ok=True)

    rows_html = ""
    for _, row in results_df.iterrows():
        color = "#ffcccc" if row['drifted'] else "#ccffcc"
        flag = "DRIFT DETECTED" if row['drifted'] else "Stable"
        rows_html += (
            f"<tr style='background:{color}'>"
            f"<td>{row['column']}</td>"
            f"<td>{row['test']}</td>"
            f"<td>{row['ref_mean']}</td>"
            f"<td>{row['cur_mean']}</td>"
            f"<td>{row['p_value']}</td>"
            f"<td><b>{flag}</b></td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Data Drift Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th {{ background-color: #4472C4; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 8px; border: 1px solid #ddd; }}
        .summary {{ background: #f0f0f0; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .ok {{ color: green; font-weight: bold; }}
        .drift {{ color: red; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Data Drift Monitoring Report</h1>
    <div class='summary'>
        <h2>Summary</h2>
        <p>Reference data: <b>{len(reference)} rows</b> (first 70% of dataset)</p>
        <p>Current data: <b>{len(current)} rows</b> (last 30% of dataset)</p>
        <p>Columns checked: <b>{len(results_df)}</b></p>
        <p>Drifted columns: <b class='{"drift" if len(drifted_cols) > 0 else "ok"}'>{len(drifted_cols)}</b></p>
        <p>Status: <b class='{"drift" if len(drifted_cols) > 2 else "ok"}'>
            {"Significant drift detected — consider retraining" if len(drifted_cols) > 2
             else "Inputs stable — no retraining needed"}
        </b></p>
    </div>
    <h2>Column-Level Drift Analysis</h2>
    <p>A p-value below 0.05 indicates statistically significant drift.</p>
    <table>
        <tr>
            <th>Column</th>
            <th>Test Used</th>
            <th>Reference Mean</th>
            <th>Current Mean</th>
            <th>p-value</th>
            <th>Status</th>
        </tr>
        {rows_html}
    </table>
    <br>
    <p><i>Generated by monitor_drift.py using KS Test (numeric) and Chi-Squared Test (categorical)</i></p>
</body>
</html>"""

    with open(DRIFT_REPORT_PATH, 'w') as f:
        f.write(html)

    print(f"\n[monitor] HTML report saved to: {DRIFT_REPORT_PATH}")
    print("[monitor] Open it in your browser to view the full report.")


if __name__ == "__main__":
    run_drift_report()