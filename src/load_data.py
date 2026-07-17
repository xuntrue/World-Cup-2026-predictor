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

# ===== CONFIG =====
ROOT = Path(__file__).resolve().parents[1]  # repo root
DATA_DIR = ROOT / "vendor" / "international_results"
VERBOSE = False # Set to True to see detailed loading/standarization output

# ===== FILSE =====
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
    if VERBOSE:
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

def count_former_names_per_column(
    df: pd.DataFrame,
    former_names_df: pd.DataFrame,
    *,
    former_col: str = "former",
    normalize: bool = True,
) -> Dict[str, int]:
    """
    Count how many records in each column of df match a former territory name.
    
    Args:
        df: DataFrame to check (e.g., results_df, goalscorers_df)
        former_names_df: DataFrame with former territory names
        former_col: Column name in former_names_df containing former names
        normalize: Whether to normalize text (case + whitespace) before matching
    
    Returns:
        Dictionary mapping column_name -> count of records with former names
    """
    result = {}
    
    # Extract unique former names
    former_names = set(former_names_df[former_col].dropna())
    
    if normalize:
        former_names = {_normalize_team_text(name) for name in former_names}
    
    # Check each column in the dataframe
    for col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
            normalized = df[col].map(lambda x: _normalize_team_text(x))
            count = (normalized.isin(former_names)).sum()
            result[col] = count
    
    return result

def count_former_names_detailed(
    df: pd.DataFrame,
    former_names_df: pd.DataFrame,
    *,
    former_col: str = "former",
    normalize: bool = True,
) -> Dict[str, Dict[str, int]]:
    """
    More detailed version: returns which former names actually appear in the data.
    
    Returns:
        Dictionary mapping column_name -> {former_name: count, ...}
    """
    result = {}
    
    # Extract unique former names
    former_names = former_names_df[former_col].dropna().unique()
    
    # Check each column in the dataframe
    for col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
            normalized = df[col].map(lambda x: _normalize_team_text(x))
            
            col_matches = {}
            for former_name in former_names:
                norm_former = _normalize_team_text(former_name) if normalize else former_name
                count = (normalized == norm_former).sum()
                if count > 0:
                    col_matches[former_name] = count
            
            result[col] = col_matches
    
    return result

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

def debug_unmapped_records(
    df: pd.DataFrame,
    former_names_df: pd.DataFrame,
    col: str = "country",
    date_col: str = "date",
    former_col: str = "former",
    start_date_col: str = "start_date",
    end_date_col: str = "end_date",
    current_col: str = "current",
) -> None:
    """
    Show which records with former names didn't get mapped and why.
    """
    former_names = set(former_names_df[former_col].dropna())
    former_names = {_normalize_team_text(name) for name in former_names}
    
    normalized = df[col].map(lambda x: _normalize_team_text(x))
    mask = normalized.isin(former_names)
    
    if not mask.any():
        print(f"No unmapped records found in {col}.")
        return
    
    unmapped = df[mask].copy()
    print(f"\nUnmapped records in {col}: {len(unmapped)}")
    
    for former_name in unmapped[col].unique():
        records = unmapped[unmapped[col] == former_name]
        print(f"\n{former_name}: {len(records)} records")
        
        # Get the date ranges for this former name from former_names_df
        norm_former = _normalize_team_text(former_name)
        ranges = former_names_df[
            former_names_df[former_col].map(_normalize_team_text) == norm_former
        ][[start_date_col, end_date_col, current_col]]
        
        print(f"  Valid ranges in former_names.csv:")
        for _, row in ranges.iterrows():
            print(f"    {row[current_col]}: {row[start_date_col]} to {row[end_date_col]}")
        
        print(f"  Actual match dates:")
        for _, rec in records.iterrows():
            print(f"    {rec[date_col]}")

def load_data():
    if VERBOSE:
        print("=== World-Cup-2026 Predictor: Data Loading ===")
        print(f"Project root: {ROOT}")
        print(f"Using data directory: {DATA_DIR}")

    #start = time.perf_counter() # code you want to time

    results_df = load_csv(files["results"])
    shootouts_df = load_csv(files["shootouts"])
    goalscorers_df = load_csv(files["goalscorers"])
    former_names_df = load_csv(files["former_names"])

    """
    counts = count_former_names_per_column(results_df, former_names_df)
    print(counts)

    detailed = count_former_names_detailed(results_df, former_names_df)
    for col, matches in detailed.items():
        if matches:
            print(f"\n{col}:")
            for former_name, count in matches.items():
                print(f"  {former_name}: {count} records")
    """

    # Toggle: set to False if you want to use current territory names.
    FORMER_NAMES = False
    USE_DATE_AWARE_MAPPING = False
    if not FORMER_NAMES:
        if VERBOSE:
            print("Starting standardization...")
        targets = [
            ("results_df", results_df, "home_team"),
            ("results_df", results_df, "away_team"),
            ("results_df", results_df, "country"),
            ("shootouts_df", shootouts_df, "home_team"),
            ("shootouts_df", shootouts_df, "away_team"),
            ("shootouts_df", shootouts_df, "winner"),
            ("shootouts_df", shootouts_df, "first_shooter"),
            ("goalscorers_df", goalscorers_df, "home_team"),
            ("goalscorers_df", goalscorers_df, "away_team"),
            ("goalscorers_df", goalscorers_df, "team"),
        ]

        for i, (df_label, df_ref, col) in enumerate(targets, start=1):
            if VERBOSE:
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
        if VERBOSE:
            if USE_DATE_AWARE_MAPPING:
                print("\nAll team names standardized using date-aware former->current mapping.")
            else:
                print("\nAll team names standardized using former->current mapping (ignoring dates).")

            # Check counts after standardization
            print("\n=== Verification ===")
            counts_after = count_former_names_per_column(results_df, former_names_df)
            print(f"Former names remaining: {counts_after}")
        
            # Show specific examples of replacements
            print("\nExample replacements in results_df:")
            detailed_after = count_former_names_detailed(results_df, former_names_df)
            for col, matches in detailed_after.items():
                if matches:
                    for former_name, count in list(matches.items())[:5]:  # First 5
                        print(f"  {former_name}: {count}")

        
            # Debug: show why some records weren't mapped
            debug_unmapped_records(results_df, former_names_df)

    #elapsed = time.perf_counter() - start
    #print(f"Elapsed: {elapsed:.4f} seconds")
    print("\nAll CSV files loaded successfully.")
    return results_df, shootouts_df, goalscorers_df, former_names_df

if __name__ == "__main__":
    load_data()