import pandas as pd
import numpy as np
import json

def normalise(column):
    """Normalize a column of data."""
    return (column - column.min()) / (column.max() - column.min())

def normalise_df(df, stats_to_norm):
    """Normalize specified columns in a dataframe."""
    for stat in stats_to_norm:
        df['NORM_{}'.format(stat)] = normalise(df[stat])
    return df

def calc_distance(a, b):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((a - b)**2))

def find_player(df, player_id, season):
    """Find a player in the dataframe based on player_id and season."""
    return df[(df['player_id'] == player_id) & (df['season'] == season)].iloc[0] if not df[(df['player_id'] == player_id) & (df['season'] == season)].empty else None

def process_data(csv_path):
    """Process and clean the input CSV data."""
    stats = ['PTS', 'MP', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'AST', 'STL', 'TOV', 'BLK', 'PF']
    df = pd.read_csv(csv_path, header=0)
    df.rename(columns={'mp_per_game': 'MP',
                    'fg_per_game': 'FGM',
                    'fga_per_game': 'FGA',
                    'x3p_per_game': 'FG3M',
                    'x3pa_per_game': 'FG3A',
                    'ft_per_game': 'FTM',
                    'fta_per_game': 'FTA',
                    'orb_per_game': 'OREB',
                    'drb_per_game': 'DREB',
                    'ast_per_game': 'AST',
                    'stl_per_game': 'STL',
                    'blk_per_game': 'BLK',
                    'tov_per_game': 'TOV',
                    'pf_per_game': 'PF',
                    'pts_per_game': 'PTS'}, inplace=True)
    df = df.drop(['seas_id','trb_per_game', 'x2pa_per_game', 'x2p_per_game'], axis=1)
    cleaned_df = df.dropna(axis=0, how='all')
    cleaned_df = cleaned_df.dropna(subset=stats)
    min_games = 10
    filtered_df = cleaned_df[cleaned_df['g'] > min_games]
    stats_to_norm = ['PTS', 'MP', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'AST', 'STL', 'TOV', 'BLK', 'PF']
    norm_df = normalise_df(filtered_df, stats_to_norm)
    return norm_df, stats

def save_processed_data(norm_df, stats, output_path):
    """Save normalized dataframe and stats to a JSON file."""
    data = {
        'normalized_data': norm_df.to_dict(orient='records'),
        'stats': stats
    }
    with open(output_path, 'w') as f:
        json.dump(data, f)

def load_processed_data(input_path):
    """Load normalized dataframe and stats from a JSON file."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    norm_df = pd.DataFrame(data['normalized_data'])
    stats = data['stats']
    return norm_df, stats
