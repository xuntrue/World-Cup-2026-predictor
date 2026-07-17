# src/main.py
"""
Load the CSV files from the vendored international_results dataset
and print clear, beginner-friendly summaries.

Run:
    python -m src.load_data
"""

from load_data import load_data
from elo_model import load_elo

# Load all dataframes
results_df, shootouts_df, goalscorers_df = load_data()

train_elo(results_df)

#train_match_model()

#train_score_model()

#train_penalty_model()

#simulate_world_cup()