def calculate_fantasy_score(player, scoring_system):
    """Calculate fantasy score for a player based on predicted stats and scoring system."""
    fantasy_points = 0
    for stat, value in scoring_system.items():
        if stat == 'REB':
            # Combine OREB and DREB for REB
            fantasy_points += (player.get('predicted_OREB', 0) + player.get('predicted_DREB', 0)) * value
        else:
            fantasy_points += player.get(f'predicted_{stat}', 0) * value
    return fantasy_points

def score_all_players(players, scoring_system):
    """Calculate fantasy scores for all players."""
    for player in players:
        player['fantasy_score'] = calculate_fantasy_score(player, scoring_system)
    return sorted(players, key=lambda x: x['fantasy_score'], reverse=True)
