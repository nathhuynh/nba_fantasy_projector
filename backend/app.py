from flask import Flask, request, jsonify
from flask_cors import CORS
from predictor import load_player_predictions
from scorer import score_all_players
from dotenv import load_dotenv
import os

load_dotenv() 

app = Flask(__name__)

CORS_ALLOWED_ORIGINS = os.getenv('CORS_ALLOWED_ORIGINS', '').split(',')
CORS(app, resources={r"/api/*": {"origins": CORS_ALLOWED_ORIGINS}}, methods=["GET", "POST", "OPTIONS"])

# Load predictions when the app starts
PREDICTIONS_PATH = 'player_predictions.json'
all_player_predictions = load_player_predictions(PREDICTIONS_PATH)

@app.route('/api/players', methods=['GET'])
def get_players():
    """Return a list of all players."""
    players = [{"id": p["player_id"], "name": p["player_name"]} for p in all_player_predictions]
    return jsonify(players)

@app.route('/api/score', methods=['POST', 'OPTIONS'])
def score_players():
    """Score players based on provided scoring system."""
    if request.method == "OPTIONS":
        return jsonify({"message": "OK"}), 200

    scoring_system = request.json.get('scoring_system')

    if not scoring_system:
        return jsonify({"error": "No scoring system provided"}), 400

    try:
        scored_players = score_all_players(all_player_predictions, scoring_system)
        return jsonify(scored_players)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/player/<player_id>', methods=['GET'])
def get_player_details(player_id):
    """Return details for a specific player."""
    # Convert player_id to int for comparison
    player_id = int(player_id)
    player = next((p for p in all_player_predictions if int(p["player_id"]) == player_id), None)
    if player:
        return jsonify(player)
    else:
        # Debug information
        app.logger.info(f"Player {player_id} not found. Available IDs: {[p['player_id'] for p in all_player_predictions[:5]]}")
        return jsonify({"error": f"Player not found. ID: {player_id}"}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
