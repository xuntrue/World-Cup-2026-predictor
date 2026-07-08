# src/load_data.py
"""
Load the CSV files from the vendored international_results dataset
and print clear, beginner-friendly summaries.

Run:
    python -m src.load_data
"""

from ast import Dict
from pathlib import Path
import pandas as pd
#import time

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

def _normalize_team_text(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

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
    Standardize team names using former_names.csv.
    If use_date_aware_mapping=True, map names based on match date being within [start_date, end_date] for that former territory.
    """

    out = df.copy()
    out[team_col] = out[team_col].map(_normalize_team_text)

    if former_names_df is None:
        return out

    required = {current_col, former_col, start_date_col, end_date_col}
    missing = required - set(former_names_df.columns)
    if missing:
        raise ValueError(f"former_names_df is missing columns: {sorted(missing)}")

    if not use_date_aware_mapping:
        # former -> current (ignores dates)
        mapping: Dict[str, str] = (
            former_names_df.dropna(subset=[former_col, current_col])
            .drop_duplicates(subset=[former_col], keep="first")
            .set_index(former_col)[current_col]
            .to_dict()
        )
        out[team_col] = out[team_col].map(lambda t: mapping.get(t, t))
        return out

    if date_col not in out.columns:
        raise ValueError(f"use_date_aware_mapping=True but '{date_col}' column not found.")

    # Parse dates once.
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    fn = former_names_df.copy()
    fn[former_col] = fn[former_col].map(_normalize_team_text)
    fn[current_col] = fn[current_col].map(_normalize_team_text)
    fn[start_date_col] = pd.to_datetime(fn[start_date_col], errors="coerce")
    fn[end_date_col] = pd.to_datetime(fn[end_date_col], errors="coerce")

    # Pre-index intervals per former name
    intervals_by_former: Dict[str, pd.DataFrame] = {}
    for former_name, g in fn.groupby(former_col, sort=False):
        intervals_by_former[former_name] = g[[start_date_col, end_date_col, current_col]].copy()

    # Build result column
    out[team_col + "__mapped"] = out[team_col]
    for team_name, idx in out.groupby(team_col).groups.items():
        if team_name == "" or team_name is None:
            continue
        if team_name not in intervals_by_former:
            continue
        match_dates = out.loc[idx, date_col]
        valid_mask = match_dates.notna()
        if not valid_mask.any():
            continue
        g_intervals = intervals_by_former[team_name]

        # Determine mapping for each date in this group by checking interval inclusion
        mapped = out.loc[idx, team_col + "__mapped"].copy()
        unmapped_mask = valid_mask

        # Treat NaT start/end as unbounded sides
        for _, row in g_intervals.iterrows():
            start_d = row[start_date_col]
            end_d = row[end_date_col]
            target = row[current_col]

            if pd.isna(start_d):
                left_ok = pd.Series(True, index=match_dates.index)
            else:
                left_ok = match_dates <= match_dates.index.map(lambda _: match_dates)  # dummy

            # Vectorized inclusion for this interval:
            if pd.isna(start_d):
                left_ok = pd.Series(True, index=match_dates.index)
            else:
                left_ok = match_dates >= start_d

            if pd.isna(end_d):
                right_ok = pd.Series(True, index=match_dates.index)
            else:
                right_ok = match_dates <= end_d

            interval_ok = left_ok & right_ok

            # Only fill still-unmapped rows
            fill_mask = unmapped_mask & interval_ok
            if fill_mask.any():
                mapped.loc[fill_mask] = target
                unmapped_mask = unmapped_mask & (~interval_ok)

            if not unmapped_mask.any():
                break

        out.loc[idx, team_col + "__mapped"] = mapped

    out[team_col] = out[team_col + "__mapped"]
    out = out.drop(columns=[team_col + "__mapped"])
    return out

def main():
    print("=== World-Cup-2026 Predictor: Data Loading ===")
    print(f"Project root: {ROOT}")
    print(f"Using data directory: {DATA_DIR}")

    results_df = load_csv(files["results"])
    shootouts_df = load_csv(files["shootouts"])
    goalscorers_df = load_csv(files["goalscorers"])

    # Toggle: set to False if you want to use current territory names.
    FORMER_NAMES = False
    USE_DATE_AWARE_MAPPING = False
    if not FORMER_NAMES:
        print("\nLoading former_names.csv...")
        former_names_df = load_csv(files["former_names"])
        print("former_names.csv loaded. Starting standardization...")
        
        #start = time.perf_counter() # code you want to time
      
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
            df_ref = standardize_team_names(
                df_ref,
                col,
                former_names_df,
                use_date_aware_mapping=USE_DATE_AWARE_MAPPING,
                date_col="date",
            )
            # Write back
            if df_ref is results_df:
                results_df = df_ref
            elif df_ref is shootouts_df:
                shootouts_df = df_ref
            elif df_ref is goalscorers_df:
                goalscorers_df = df_ref
        print("\nAll team names standardized.")
        
        #elapsed = time.perf_counter() - start
        #print(f"Elapsed: {elapsed:.3f} seconds")

    print("\nAll CSV files loaded successfully.")

if __name__ == "__main__":
    main()