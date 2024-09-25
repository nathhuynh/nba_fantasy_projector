import requests

BASE_URL = 'http://localhost:5000'

def test_hello():
    response = requests.get(f'{BASE_URL}/api/hello')
    print('Hello endpoint:', response.json())

def test_get_players():
    response = requests.get(f'{BASE_URL}/api/players')
    print('Players endpoint:', response.json()[:5])  # Print first 5 players

def test_score_players():
    scoring_system = {"PTS": 1, "AST": 2, "REB": 1}
    response = requests.post(f'{BASE_URL}/api/score', json={'scoring_system': scoring_system})
    print('Score endpoint:', response.json()[:5])  # Print first 5 scored players

def test_get_player_details(player_id):
    response = requests.get(f'{BASE_URL}/api/player/{player_id}')
    print(f'Player {player_id} details:', response.json())

if __name__ == '__main__':
    test_hello()
    test_get_players()
    test_score_players()
    
    # Get a player ID from the get_players response
    players_response = requests.get(f'{BASE_URL}/api/players')
    if players_response.json():
        first_player_id = players_response.json()[0]['id']
        test_get_player_details(first_player_id)
