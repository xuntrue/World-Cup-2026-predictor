# src/elo_model.py
"""
Elo rating system for international football matches.
Organized into classes for better maintainability and modularity.
"""

from load_data import load_data

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pandas as pd

# Load all dataframes
results_df, _, _, _ = load_data()

# ===== Elo config =====
INITIAL_ELO = 1200
K_FACTOR = 40  # Standard K-factor = 32 (higher means more volatile ratings)
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

# === File Handling ===
def save_elo_results(team_ratings: dict,
                     rating_history: list, 
                     ratings_file: str = "data/team_elo.csv",
                     history_file: str = "data/rating_history.csv") -> None:
    """
    Save Elo training results to CSV files.
    """
    # Save team ratings
    ratings_df = pd.DataFrame([
        {"team": team, "elo_rating": elo}
        for team, elo in sorted(team_ratings.items())
    ])
    ratings_df.to_csv(ratings_file, index=False)
    print(f"✓ Saved {len(ratings_df)} team ratings to '{ratings_file}'")
    
    # Save match history
    history_df = pd.DataFrame(rating_history)
    history_df.to_csv(history_file, index=False)
    print(f"✓ Saved {len(history_df)} match records to '{history_file}'")

def load_elo_results(ratings_file: str = "data/rating_history.csv",
                     history_file: str = "data/rating_history.csv") -> tuple:
    """
    Load cached Elo results from CSV files.
    
    Returns:
        (team_ratings dict, rating_history DataFrame)
        Returns (None, None) if files don't exist.
    """
    if not os.path.exists(ratings_file) or not os.path.exists(history_file):
        return None, None
    
    # Load team ratings
    ratings_df = pd.read_csv(ratings_file)
    team_ratings = dict(zip(ratings_df['team'], ratings_df['elo_rating']))
    
    # Load history
    history_df = pd.read_csv(history_file)
    
    print(f"✓ Loaded {len(team_ratings)} team ratings from '{ratings_file}'")
    print(f"✓ Loaded {len(history_df)} match records from '{history_file}'")
    
    return team_ratings, history_df

def create_team_timeline(rating_history: list) -> pd.DataFrame:
    """
    Extract team Elo ratings at each point in time.
    Creates a long-format DataFrame: one row per team per match date.
    """
    timeline_records = []
    
    for match in rating_history:
        # Home team snapshot
        timeline_records.append({
            "date": match["date"],
            "team": match["home_team"],
            "elo_rating": match["home_elo_after"],  # Rating after this match
        })
        
        # Away team snapshot
        timeline_records.append({
            "date": match["date"],
            "team": match["away_team"],
            "elo_rating": match["away_elo_after"],  # Rating after this match
        })
    
    timeline_df = pd.DataFrame(timeline_records)
    timeline_df = timeline_df.sort_values(["team", "date"]).reset_index(drop=True)
    
    return timeline_df

def save_team_timeline(rating_history: list, timeline_file: str = "data/team_timeline.csv") -> None:
    """
    Save team Elo timeline to CSV.
    """
    timeline_df = create_team_timeline(rating_history)
    timeline_df.to_csv(timeline_file, index=False)
    print(f"✓ Saved team timeline ({len(timeline_df)} records) to '{timeline_file}'")

# === Analysis ===
def analyze_elo_distribution(team_ratings: dict) -> dict:
    """
    Calculate descriptive statistics for Elo ratings.
    
    Returns:
        Dictionary with statistics (mean, median, std dev, percentiles, etc.)
    """
    ratings = np.array(list(team_ratings.values()))
    
    stats = {
        "count": len(ratings),
        "mean": np.mean(ratings),
        "median": np.median(ratings),
        "std_dev": np.std(ratings),
        "min": np.min(ratings),
        "max": np.max(ratings),
        "q25": np.percentile(ratings, 25),
        "q75": np.percentile(ratings, 75),
        "range": np.max(ratings) - np.min(ratings),
    }
    
    return stats

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

def print_elo_statistics(stats: dict) -> None:
    """
    Pretty-print Elo distribution statistics.
    """
    print("\n" + "="*60)
    print("ELO DISTRIBUTION STATISTICS")
    print("="*60)
    print(f"Teams analyzed:          {stats['count']}")
    print(f"Mean Elo:                {stats['mean']:.1f}")
    print(f"Median Elo:              {stats['median']:.1f}")
    print(f"Std. Deviation:          {stats['std_dev']:.1f}")
    print(f"Min Elo:                 {stats['min']:.1f}")
    print(f"Max Elo:                 {stats['max']:.1f}")
    print(f"Range (Max - Min):       {stats['range']:.1f}")
    print(f"25th percentile:         {stats['q25']:.1f}")
    print(f"75th percentile:         {stats['q75']:.1f}")
    print(f"Interquartile range:     {stats['q75'] - stats['q25']:.1f}")
    print("="*60)

# === Visualisation ===
def plot_elo_distribution(team_ratings: dict, stats: dict, save: bool) -> None:
    """
    Visualize Elo rating distribution with histogram and statistics.
    """
    ratings = np.array(list(team_ratings.values()))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- LEFT: Histogram with overlays ---
    ax1 = axes[0]
    ax1.hist(ratings, bins=60, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Add vertical lines for key statistics
    ax1.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.0f}")
    ax1.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.0f}")
    ax1.axvline(INITIAL_ELO, color='orange', linestyle=':', linewidth=2, label=f"Initial Elo: {INITIAL_ELO}")
    
    ax1.set_xlabel('Elo Rating', fontsize=11)
    ax1.set_ylabel('Number of Teams', fontsize=11)
    ax1.set_title('Elo Rating Distribution (Histogram)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # --- RIGHT: Box plot ---
    ax2 = axes[1]
    bp = ax2.boxplot(ratings, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    
    # Add scatter of actual data points
    ax2.scatter([1] * len(ratings), ratings, alpha=0.4, s=50, color='steelblue')
    
    ax2.set_ylabel('Elo Rating', fontsize=11)
    ax2.set_title('Elo Rating Distribution (Box Plot)', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(['All Teams'])
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save:
        plt.savefig('elo_distribution.png', dpi=150, bbox_inches='tight')
        print("\n✓ Distribution plot saved to 'elo_distribution.png'")
    plt.show()

def plot_team_elo_history(team_name: str, timeline_file: str = "team_timeline.csv", save: bool = False) -> None:
    """
    Plot Elo rating history for a specific team over time.
    """
    timeline_df = pd.read_csv(timeline_file)
    team_data = timeline_df[timeline_df['team'] == team_name].sort_values('date')
    
    if len(team_data) == 0:
        print(f"Team '{team_name}' not found in timeline.")
        return
    
    plt.figure(figsize=(14, 6))
    plt.plot(pd.to_datetime(team_data['date']), team_data['elo_rating'], 
             linewidth=2, marker='o', markersize=3, alpha=0.7, color='steelblue')
    
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Elo Rating', fontsize=11)
    plt.title(f'{team_name} Elo Rating Over Time', fontsize=13, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save:
        plt.savefig(f'{team_name.lower()}_elo_history.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to '{team_name.lower()}_elo_history.png'")
    plt.show()

def plot_multiple_teams_elo_history(team_names: list, timeline_file: str = "data/team_timeline.csv", save: bool = False) -> None:
    """
    Plot Elo rating history for multiple teams on the same graph.
    """
    timeline_df = pd.read_csv(timeline_file)
    
    plt.figure(figsize=(16, 7))
    
    for team in team_names:
        team_data = timeline_df[timeline_df['team'] == team].sort_values('date')
        if len(team_data) > 0:
            plt.plot(pd.to_datetime(team_data['date']), team_data['elo_rating'], 
                     linewidth=2, marker='o', markersize=2, alpha=0.7, label=team)
    
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Elo Rating', fontsize=11)
    plt.title('Elo Rating History - Multiple Teams', fontsize=13, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save:
        plt.savefig('teams_elo_history.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to 'teams_elo_history.png'")
    plt.show()

if __name__ == "__main__":
    #team_ratings, history = train_elo(results_df)
    #save_elo_results(team_ratings, history)
    team_ratings, history = load_elo_results()

    history_df = pd.DataFrame(history)

    save_team_timeline(history_df.to_dict('records')) 

    if VERBOSE:
        # Get a specific team's current rating
        #france_rating = get_team_rating('FRANCE', team_ratings)
        #print(f"France Elo: {france_rating:.1f}") #prints France Elo: nan

        # View match history
        #print(history_df.head())
        #print(history_df.tail())

        # Print in Elo order (descending)
        print("\n=== Teams Sorted by Elo (Highest to Lowest) ===")
        print_ratings_by_elo(team_ratings)
    
        # Print in alphabetical order
        #print("\n=== Teams Sorted Alphabetically ===")
        #print_ratings_alphabetically(team_ratings)
    
        # Analyze distribution
        stats = analyze_elo_distribution(team_ratings)
        print_elo_statistics(stats)
        plot_elo_distribution(team_ratings, stats, save=False)

         # Plot individual team histories
        print("\n=== Team Elo History Plots ===")
        #plot_team_elo_history('FRANCE', save=False)
        #plot_team_elo_history('SPAIN', save=False)
        #plot_team_elo_history('ARGENTINA', save=False)
    
        # Plot multiple teams together
        plot_multiple_teams_elo_history(['FRANCE', 'SPAIN', 'BRAZIL', 'GERMANY', 'ARGENTINA'], save=False)