# src/elo_model.py
"""
Load the CSV files from the vendored international_results dataset
and print clear, beginner-friendly summaries.

Run:
    python -m src.load_data
"""

from load_data import load_data

import math
import pandas as pd

# Load all dataframes
results_df, _, _, _ = load_data()

# ===== Elo config =====
INITIAL_ELO = 1000
K_FACTOR = 64  # Standard K-factor = 32 (higher means more volatile ratings)
VERBOSE = True  # Set to False to suppress match-by-match output

def train_elo(results_df):
    """
    Train Elo ratings for all teams based on historical match results.
    Processes matches serially in chronological order.
    
    Returns:
        Dictionary mapping team -> current Elo rating
        DataFrame with match-by-match rating history (optional)
    """
    
    teams = set()
    teams.update(results_df["home_team"].unique())
    teams.update(results_df["away_team"].unique())
    
    # Store current ratings
    team_ratings = {team: INITIAL_ELO for team in teams}
    
    # Sort matches chronologically
    matches = results_df.sort_values("date").reset_index(drop=True)
    
    # Track history (optional but useful for debugging/analysis)
    rating_history = []
    
    # Process each match
    for idx, match in matches.iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]
        home_score = match["home_score"]
        away_score = match["away_score"]
        match_date = match["date"]

        # Get current ratings
        home_elo = team_ratings[home_team]
        away_elo = team_ratings[away_team]
        
        if VERBOSE:
            print(f"Procesing Match #{idx}/{len(results_df)} {match_date} {home_team}[{round(home_elo)}] vs. {away_team}[{round(away_elo)}]...")

        # Calculate expected scores
        home_expected = expected_score(home_elo, away_elo)
        away_expected = expected_score(away_elo, home_elo)
        
        # Determine actual result (1 = win, 0.5 = draw, 0 = loss)
        if home_score > away_score:
            home_actual, away_actual = 1.0, 0.0
        elif home_score < away_score:
            home_actual, away_actual = 0.0, 1.0
        else:
            home_actual, away_actual = 0.5, 0.5
        
        # Calculate goal difference multiplier (optional: amplify rating change for blowouts)
        goal_diff = abs(home_score - away_score)
        goal_multiplier = 1 + (goal_diff * 0.05)  # Each goal adds 5% more change
        
        # Calculate rating changes
        home_change = K_FACTOR * goal_multiplier * (home_actual - home_expected)
        away_change = K_FACTOR * goal_multiplier * (away_actual - away_expected)
        
        # Update ratings
        team_ratings[home_team] += home_change
        team_ratings[away_team] += away_change
        
        # Store history
        rating_history.append({
            "date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "home_elo_before": home_elo,
            "away_elo_before": away_elo,
            "home_elo_after": team_ratings[home_team],
            "away_elo_after": team_ratings[away_team],
        })
    
    if VERBOSE:
        print(f"\nElo training complete. Processed {len(matches)} matches.")
        print(f"Trained {len(team_ratings)} teams.")
    
        # DEBUG: Check what we're about to return
        print("\n=== DEBUG: team_ratings before return ===")
        print(f"Type: {type(team_ratings)}")
        print(f"Length: {len(team_ratings)}")
    
        france_val = team_ratings.get('FRANCE', 'KEY_NOT_FOUND')
        print(f"FRANCE value: {france_val}")
        print(f"FRANCE is NaN? {math.isnan(france_val) if isinstance(france_val, float) else 'N/A'}")
    
        # Sample a few entries
        print("\nFirst 5 team_ratings entries:")
        for i, (team, rating) in enumerate(list(team_ratings.items())[:5]):
            print(f"  {team}: {rating} (type: {type(rating).__name__})")

    return team_ratings, rating_history

def expected_score(elo_a: float, elo_b: float) -> float:
    """
    Calculate expected win probability for team A using Elo formula.
    """
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def get_team_rating(team_name: str, team_ratings: dict) -> float:
    """
    Get a team's current Elo rating.
    """
    return team_ratings.get(team_name, team_ratings)

def print_ratings_by_elo(team_ratings: dict) -> None:
    """
    Print all team ratings sorted by Elo (highest to lowest).
    Filters out NaN values and reports them separately.
    """
    valid_teams = {}
    nan_teams = []
    
    for team, elo in team_ratings.items():
        if math.isnan(elo) if isinstance(elo, float) else False:
            nan_teams.append(team)
        else:
            valid_teams[team] = elo
    
    sorted_teams = sorted(valid_teams.items(), key=lambda x: x[1], reverse=True)
    for rank, (team, elo) in enumerate(sorted_teams, start=1):
        print(f"{rank:3d}. {team:30s} {round(elo):5d}")
    
    if nan_teams:
        print(f"\n⚠️  WARNING: {len(nan_teams)} teams have NaN ratings:")
        for team in sorted(nan_teams):
            print(f"  - {team}")


def print_ratings_alphabetically(team_ratings: dict) -> None:
    """
    Print all team ratings sorted alphabetically by team name.
    Filters out NaN values.
    """
    valid_teams = {team: elo for team, elo in team_ratings.items() 
                   if not (math.isnan(elo) if isinstance(elo, float) else False)}
    
    sorted_teams = sorted(valid_teams.items(), key=lambda x: x[0])
    for team, elo in sorted_teams:
        print(f"{team:30s} {round(elo):5d}")

if __name__ == "__main__":
    team_ratings, history = train_elo(results_df)
    
    # Print in Elo order (descending)
    print("\n=== Teams Sorted by Elo (Highest to Lowest) ===")
    print_ratings_by_elo(team_ratings)
    
    # Print in alphabetical order
    print("\n=== Teams Sorted Alphabetically ===")
    print_ratings_alphabetically(team_ratings)
    
    # Get a specific team's current rating
    france_rating = get_team_rating('FRANCE', team_ratings)
    print(f"France Elo: {france_rating:.1f}") #prints France Elo: nan
    
    # View match history
    history_df = pd.DataFrame(history)
    print(history_df.head())