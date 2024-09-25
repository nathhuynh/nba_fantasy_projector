from scorer import score_all_players
from predictor import load_player_predictions

def main(predictions_path, scoring_system):
    # Load pre-processed predictions
    all_player_predictions = load_player_predictions(predictions_path)

    # Calculate fantasy scores
    scored_players = score_all_players(all_player_predictions, scoring_system)

    return scored_players

if __name__ == "__main__":
    predictions_path = 'player_predictions.json'
    
    # WEB-APP: This should be adjustable by the User on the web app UI
    scoring_system = {
        'PTS': 1, 'FGM': 2, 'FGA': -1, 'FTM': 1, 'FTA': -1, 'FG3M': 1,
        'OREB': 1, 'DREB': 1, 'AST': 2, 'STL': 4, 'BLK': 4, 'TOV': -2
    }

    results = main(predictions_path, scoring_system)
    
    # Print top 10 players
    for player in results[:10]:
        print(f"{player['player_name']}: {player['fantasy_score']:.2f}")
