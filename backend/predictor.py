import numpy as np
from data_processor import calc_distance, find_player
import json

def stats_prediction(df, current_season, current_player_id, stats):
    """Predict stats for a player based on historical data."""
    if not ((df['season'] == current_season) & (df['player_id'] == current_player_id)).any():
        print(f'Cannot find player in season {current_season} with player_id {current_player_id}')
        return None

    distance_list = []
    current_player_stats = np.array([
        df.loc[(df['player_id'] == current_player_id) & (df['season'] == current_season), f'NORM_{stat}'].values[0]
        for stat in ['PTS', 'MP', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'AST', 'STL', 'TOV', 'BLK', 'PF']
    ])
    
    for row in df.itertuples():
        target_player_stats = np.array([
            getattr(row, f'NORM_{stat}') for stat in 
            ['PTS', 'MP', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'AST', 'STL', 'TOV', 'BLK', 'PF']
        ])

        distance = calc_distance(current_player_stats, target_player_stats)
        distance_list.append(distance)
    
    df = df.copy()
    df['distance'] = distance_list
    sorted_df = df.sort_values('distance')

    current_player_data = df.loc[(df['season'] == current_season) & (df['player_id'] == current_player_id)].iloc[0]
    current_player_name = current_player_data['player']
    current_player_position = current_player_data['pos']

    predicted_stats = {
        'player_id': current_player_id,
        'player_name': current_player_name,
        'position': current_player_position
    }
    print(f'Predicting {current_season + 1} stats for player: {current_player_name} (player id: {current_player_id}, position: {current_player_position})')
    
    for col in stats:
        stat_sum = weight_sum = 0

        for idx, row in sorted_df.iloc[0:21].iterrows():
            if row.season == current_season or row.distance == 0:
                continue

            weight = 1 / row.distance
            following_season = row.season + 1
            following_season_stats = find_player(sorted_df, row.player_id, following_season)

            if following_season_stats is None:
                continue
            
            stat_sum += getattr(following_season_stats, col) * weight
            weight_sum += weight
            
        if weight_sum != 0:
            predicted_stats[f'predicted_{col}'] = stat_sum / weight_sum
    
    return predicted_stats

def predict_all_players(df, current_season, stats):
    """Predict stats for all players in the given season."""
    df_current_season = df[df['season'] == current_season]
    player_ids = df_current_season['player_id'].unique()

    all_player_predictions = []
    for player_id in player_ids:
        prediction = stats_prediction(df, current_season, player_id, stats)
        if prediction:
            all_player_predictions.append(prediction)

    return all_player_predictions

def predict_and_save_all_players(df, current_season, stats, output_path):
    """Predict stats for all players and save results to a JSON file."""
    all_player_predictions = predict_all_players(df, current_season, stats)
    
    # Convert numpy types to native Python types
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Apply the conversion to each player's predictions
    serializable_predictions = []
    for player in all_player_predictions:
        serializable_player = {k: convert_to_serializable(v) for k, v in player.items()}
        serializable_predictions.append(serializable_player)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_predictions, f)
    
    return all_player_predictions

def load_player_predictions(input_path):
    """Load player predictions from a JSON file."""
    with open(input_path, 'r') as f:
        all_player_predictions = json.load(f)
    return all_player_predictions
