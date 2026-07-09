# src/load_data.py
"""
Load the CSV files from the vendored international_results dataset
and print clear, beginner-friendly summaries.

Run:
    python -m src.load_data
"""

from typing import Optional, Dict
from pathlib import Path
import pandas as pd
import re
import time

ROOT = Path(__file__).resolve().parents[1]  # repo root
DATA_DIR = ROOT / "vendor" / "international_results"

files = {
    "results": DATA_DIR / "results.csv",
    "shootouts": DATA_DIR / "shootouts.csv",
    "goalscorers": DATA_DIR / "goalscorers.csv",
    "former_names": DATA_DIR / "former_names.csv"
}

def load_csv(path: Path) -> pd.DataFrame:
    # Check that the file exists
    if not path.exists(): 
        raise FileNotFoundError(f"Missing file: {path}")

    # Read the CSV into a pandas DataFrame.
    df = pd.read_csv(path)

    # Toggle: print a friendly summary
    if False:
        print(f"\n=== Loaded: {path} ===")
        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")
        print("Column names:")
        for col in df.columns:
            print(f"  - {col}")

        print("\nPreview (first 5 rows):")
        print(df.head(5))

    return df

def print_value_counts(
    df: "pd.DataFrame",
    col: str, *,
    normalize: bool = False,
    dropna: bool = False,
    topk: int | None = None
) -> None:
    counts = df[col].value_counts(dropna=dropna, normalize=normalize)
    if topk is not None:
        counts = counts.head(topk)
    print(counts.to_string())

def print_all_columns_value_counts(
    df: pd.DataFrame, *,
    normalize: bool = False,
    dropna: bool = False,
    topk: int | None = None
) -> None:
    for col in df.columns:
        print_value_counts(df, col, normalize=normalize, dropna=dropna, topk=topk)

def _normalize_team_text(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)  # collapse internal whitespace
    return s.upper()            # normalize case

def standardize_team_names(
    df: pd.DataFrame,
    team_col: str,
    former_names_df: Optional[pd.DataFrame],
    *,
    use_date_aware_mapping: bool = True,
    date_col: str = "date",
    current_col: str = "current",
    former_col: str = "former",
    start_date_col: str = "start_date",
    end_date_col: str = "end_date",
) -> pd.DataFrame:
    """
    Standardize team names using former_names_df.

    - Non-date-aware: map former -> current regardless of date.
    - Date-aware: map former -> current only when match date is within [start_date, end_date]
      for that former entry.
    """

    out = df.copy()

    if former_names_df is None:
        out[team_col] = out[team_col].map(_normalize_team_text)
        return out

    required = {current_col, former_col, start_date_col, end_date_col}
    missing = required - set(former_names_df.columns)
    if missing:
        raise ValueError(f"former_names_df is missing columns: {sorted(missing)}")

    # Normalize inputs
    out[team_col] = out[team_col].map(_normalize_team_text)
    fn = former_names_df.copy()
    fn[former_col] = fn[former_col].map(_normalize_team_text)
    fn[current_col] = fn[current_col].map(_normalize_team_text)

    # Ensure match dates are datetime (only used in date-aware mode)
    if use_date_aware_mapping:
        if date_col not in out.columns:
            raise ValueError(f"use_date_aware_mapping=True but '{date_col}' column not found.")
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        dates = out[date_col]

        fn[start_date_col] = pd.to_datetime(fn[start_date_col], errors="coerce")
        fn[end_date_col] = pd.to_datetime(fn[end_date_col], errors="coerce")
        # handle open-ended end_date
        fn[end_date_col] = fn[end_date_col].fillna(pd.Timestamp.max)

        # Build mapped column
        out["_team_mapped"] = out[team_col]

        # Loop per former name (no per-row loops); inside each former, loop intervals only
        # (usually small). If overlaps exist, later intervals (sorted by start_date) overwrite.
        base = out[team_col]

        for former_name, g in fn.groupby(former_col, sort=False):
            if not former_name:
                continue

            idx = base.eq(former_name)
            if not idx.any():
                continue

            # Only consider rows where team == former_name and match date is valid
            rows_idx = out.index[idx]
            sub_idx = rows_idx[dates.loc[rows_idx].notna()]
            if len(sub_idx) == 0:
                continue

            g = g[[start_date_col, end_date_col, current_col]].copy()
            g = g.dropna(subset=[start_date_col, end_date_col])
            if g.empty:
                continue

            # Deterministic overwrite rule: later start_date wins
            g = g.sort_values([start_date_col, end_date_col], kind="mergesort")

            sub_dates = dates.loc[sub_idx].to_numpy()

            mapped = out.loc[sub_idx, "_team_mapped"].to_numpy()

            # Apply each interval vectorized for this former group
            for _, row in g.iterrows():
                s = row[start_date_col]
                e = row[end_date_col]
                target = row[current_col]

                ok = (sub_dates >= s) & (sub_dates <= e)
                if ok.any():
                    mapped[ok] = target

            out.loc[sub_idx, "_team_mapped"] = mapped

        out[team_col] = out["_team_mapped"]
        out = out.drop(columns=["_team_mapped"], errors="ignore")
        return out

    # Non-date-aware mapping
    mapping_df = (
        fn.dropna(subset=[former_col, current_col])
          .drop_duplicates(subset=[former_col], keep="first")
          .set_index(former_col)[current_col]
    )
    out[team_col] = out[team_col].map(mapping_df).fillna(out[team_col])
    return out

def main():
    print("=== World-Cup-2026 Predictor: Data Loading ===")
    print(f"Project root: {ROOT}")
    print(f"Using data directory: {DATA_DIR}")

    results_df = load_csv(files["results"])
    shootouts_df = load_csv(files["shootouts"])
    goalscorers_df = load_csv(files["goalscorers"])
    former_names_df = load_csv(files["former_names"])

    # Toggle: set to False if you want to use current territory names.
    FORMER_NAMES = False
    USE_DATE_AWARE_MAPPING = True
    if not FORMER_NAMES:
        print("\nLoading former_names.csv...")
        former_names_df = load_csv(files["former_names"])
        print("former_names.csv loaded. Starting standardization...")

        start = time.perf_counter() # code you want to time
      
        targets = [
            ("results_df", results_df, "home_team"),
            ("results_df", results_df, "away_team"),
            ("shootouts_df", shootouts_df, "home_team"),
            ("shootouts_df", shootouts_df, "away_team"),
            ("goalscorers_df", goalscorers_df, "home_team"),
            ("goalscorers_df", goalscorers_df, "away_team"),
            ("goalscorers_df", goalscorers_df, "team"),
        ]

        for i, (df_label, df_ref, col) in enumerate(targets, start=1):
            print(f"\n[{i}/{len(targets)}] Standardizing {df_label}:{col} (rows={len(df_ref):,})...")
            if df_label == "results_df":
                results_df = standardize_team_names(
                    results_df, col, former_names_df,
                    use_date_aware_mapping=USE_DATE_AWARE_MAPPING,
                    date_col="date",
                )
            elif df_label == "shootouts_df":
                shootouts_df = standardize_team_names(
                    shootouts_df, col, former_names_df,
                    use_date_aware_mapping=USE_DATE_AWARE_MAPPING,
                    date_col="date",
                )
            elif df_label == "goalscorers_df":
                goalscorers_df = standardize_team_names(
                    goalscorers_df, col, former_names_df,
                    use_date_aware_mapping=USE_DATE_AWARE_MAPPING,
                    date_col="date",
                )
        if USE_DATE_AWARE_MAPPING:
            print("\nAll team names standardized using date-aware former->current mapping.")
        else:
            print("\nAll team names standardized using former->current mapping (ignoring dates).")

        
        elapsed = time.perf_counter() - start
        print(f"Elapsed: {elapsed:.4f} seconds")

    print("\nAll CSV files loaded successfully.")

if __name__ == "__main__":
    main()