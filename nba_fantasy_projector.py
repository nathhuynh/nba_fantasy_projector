import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None 

# Normalising stats
def normalise(column):
    return (column - column.min()) / (column.max() - column.min())

def normalise_df(df, stats_to_norm):
    for stat in stats_to_norm:
        df['NORM_{}'.format(stat)] = normalise(df[stat])
    return df

# Shortest path between two points
def calc_distance(a, b):
    dist = np.sqrt(np.sum(a - b)**2)
    return dist

# Finding a player in the df
def find_player(df, player_id, season):
    for row in df.itertuples():
        if player_id == row.player_id and season == row.season:
            return row

# Predicting the stats of all players for the 2023-24 season
def stats_prediction(df, current_season, current_player_id, stats):
    if not ((df['season'] == current_season) & (df['player_id'] == current_player_id)).any():
        print('Cannot find player in season {} with player_id {}'.format(current_season, current_player_id))
        return

    distance_list = []
    # Calculate all of current player's normalised stats, using the normalised dataframe that has been grouped by season
    current_player_stats = np.array([
        (df.loc[(df['player_id'] == current_player_id) & (df['season'] == current_season), 'NORM_PTS']).values[0],
        (df.loc[(df['player_id'] == current_player_id) & (df['season'] == current_season), 'NORM_MP']).values[0],
        (df.loc[(df['player_id'] == current_player_id) & (df['season'] == current_season), 'NORM_FGM']).values[0],
        (df.loc[(df['player_id'] == current_player_id) & (df['season'] == current_season), 'NORM_FGA']).values[0],
        (df.loc[(df['player_id'] == current_player_id) & (df['season'] == current_season), 'NORM_FG3M']).values[0],
        (df.loc[(df['player_id'] == current_player_id) & (df['season'] == current_season), 'NORM_FG3A']).values[0],
        (df.loc[(df['player_id'] == current_player_id) & (df['season'] == current_season), 'NORM_FTM']).values[0],
        (df.loc[(df['player_id'] == current_player_id) & (df['season'] == current_season), 'NORM_FTA']).values[0],
        (df.loc[(df['player_id'] == current_player_id) & (df['season'] == current_season), 'NORM_OREB']).values[0],
        (df.loc[(df['player_id'] == current_player_id) & (df['season'] == current_season), 'NORM_DREB']).values[0],
        (df.loc[(df['player_id'] == current_player_id) & (df['season'] == current_season), 'NORM_AST']).values[0],
        (df.loc[(df['player_id'] == current_player_id) & (df['season'] == current_season), 'NORM_STL']).values[0],
        (df.loc[(df['player_id'] == current_player_id) & (df['season'] == current_season), 'NORM_TOV']).values[0],
        (df.loc[(df['player_id'] == current_player_id) & (df['season'] == current_season), 'NORM_BLK']).values[0],
        (df.loc[(df['player_id'] == current_player_id) & (df['season'] == current_season), 'NORM_PF']).values[0]
    ])
    
    # For each row, create a vector for all normalised stats for the target player
    for row in df.itertuples():
        target_player_stats = np.array([
            row.NORM_PTS,
            row.NORM_MP,
            row.NORM_FGM,
            row.NORM_FGA,
            row.NORM_FG3M,
            row.NORM_FG3A,
            row.NORM_FTM,
            row.NORM_FTA,
            row.NORM_OREB,
            row.NORM_DREB,
            row.NORM_AST,
            row.NORM_STL,
            row.NORM_TOV,
            row.NORM_BLK,
            row.NORM_PF
        ])

        vectorized_func = np.vectorize(calc_distance)
        distance = vectorized_func(current_player_stats, target_player_stats)
        diff = np.sum(np.abs(distance) / len(distance))
        distance_list.append(diff)
    
    df['distance'] = distance_list
    
    # Dataframe sorted by shortest distance
    sorted_df = df.sort_values('distance')

    # Extract current player's name
    current_player_name = df.loc[(df['season'] == current_season) & (df['player_id'] == current_player_id), 'player'].iloc[0]

    predicted_stats = {}
    print('Predicting 2023-24 stats for player: {} (player id: {}) '.format(current_player_name, current_player_id))
    
    for col in stats:
        stat_sum = 0
        weight_sum = 0

        # Compare the target player with the top 20 most similar player seasons in history
        for idx, row in sorted_df.iloc[0:21].iterrows():
            if col == 'PTS':
                print('Comparing with {} in {}' .format(row.player, row.season))
            # Can't take the following season, skip it
            if row.season == 2023:
                continue

            # Player season is being compared with itself, skip it
            if row.distance == 0:
                continue

            weight = (1 / row.distance)
            following_season = row.season + 1

            # Use the find player function to find the stats for the following season
            following_season_stats = find_player(sorted_df, row.player_id, following_season)

            # Specific player never played in the following season (i.e. retired, injured, etc)
            if following_season_stats == None:
                continue
            
            stat_sum += getattr(following_season_stats, col) * weight
            weight_sum += weight
            
        if weight_sum != 0:
            # Using player_id as an identifier in the predicted stats dictionary
            predicted_stats['player_id'] = current_player_id

            # Using the season being predicted as an identifier
            predicted_stats['predicted_season'] = following_season

            # Calculate the predicted value for each stat column
            predicted_stats['predicted_' + col] = (stat_sum / weight_sum)
    
    return predicted_stats

scoring_system = {
    'PTS': 1,
    'FGM': 2,
    'FGA': -1,
    'FTM': 1,
    'FTA': -1,
    'FG3M': 1,
    'OREB': 1,
    'DREB': 1,
    'AST': 2,
    'STL': 4,
    'BLK': 4,
    'TOV': -2
}

def calculate_fantasy_score(player):
    fantasy_points = (
        player['predicted_PTS'] * scoring_system['PTS'] +
        player['predicted_FGM'] * scoring_system['FGM'] +
        player['predicted_FGA'] * scoring_system['FGA'] +
        player['predicted_FTM'] * scoring_system['FTM'] +
        player['predicted_FTA'] * scoring_system['FTA'] +
        player['predicted_FG3M'] * scoring_system['FG3M'] +
        player['predicted_OREB'] * scoring_system['OREB'] +
        player['predicted_DREB'] * scoring_system['DREB'] +
        player['predicted_AST'] * scoring_system['AST'] +
        player['predicted_STL'] * scoring_system['STL'] +
        player['predicted_BLK'] * scoring_system['BLK'] +
        player['predicted_TOV'] * scoring_system['TOV']
    )
    
    return fantasy_points

def main():
    # Cleaning the data
    stats = ['PTS', 'MP', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'AST', 'STL', 'TOV', 'BLK', 'PF']
    df = pd.read_csv('D:/nba_fantasy_projector/stats-csv/nba-per-game-stats.csv', header = 0)
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

    # Generating stats predictions for each player
    norm_df = normalise_df(filtered_df, stats_to_norm)
    df_2023 = norm_df[norm_df['season'] == 2023]
    final_df = df_2023.drop_duplicates(subset='player_id')
    player_ids_2023 = final_df['player_id'].tolist()

    all_player_predictions = []
    for player in player_ids_2023:
        prediction = stats_prediction(filtered_df, 2023, player, stats)
        if (prediction == None):
            continue
        all_player_predictions.append(prediction)

    # Calculate projected fantasy points for each player and inserting into the final df
    for player in all_player_predictions:
        if not player:
            continue
        idx = final_df[final_df['player_id'] == player['player_id']].index
        predicted_fantasy_points = calculate_fantasy_score(player)

        # Calculating average and total projected fantasy points for each player's season, assuming all players will play 70 games
        final_df.loc[idx, 'AVG'] = predicted_fantasy_points
        final_df.loc[idx, 'FPTS'] = predicted_fantasy_points * 70


    # Sorting the final df by total projected fantasy points
    final_df = final_df.sort_values(by='FPTS', ascending=False)
    # Removing columns that aren't neccesary for to export to reduce clutter
    stats.append('AVG')
    stats.append('FPTS')
    stats.append('player')
    final_df = final_df.loc[:, stats]
    final_df = final_df[ ['player'] + [ col for col in final_df.columns if col != 'player' ]]
    final_df.to_csv('D:/nba_fantasy_projector/nba_fantasy_projections_2023-24.csv', index=False)

if __name__ == "__main__":
    main()