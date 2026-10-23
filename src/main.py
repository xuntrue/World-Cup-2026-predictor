from data.load_data import load_data
from data.analyze_data import analyze_data

#from elo_model import  load_elo_results

def main():
    # Load all dataframes
    results_df, shootouts_df, goalscorers_df, former_names_df = load_data()
    analyze_data() # Also called load_data()

    # Load all dataframes
    #results_df, shootouts_df, goalscorers_df = load_data()

    # Get team elo ratings
    #team_ratings, _ = load_elo_results()

    #train_match_model()

    #train_score_model()

    #train_penalty_model()

    #simulate_world_cup()

if __name__ == "__main__":
    main()