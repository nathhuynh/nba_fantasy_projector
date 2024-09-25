import json
from data_processor import process_data, save_processed_data
from predictor import predict_and_save_all_players

def main():
    csv_path = 'stats-csv/nba-per-game-stats.csv'
    current_season = 2023
    processed_data_path = 'processed_data.json'
    predictions_path = 'player_predictions.json'

    # Process and save data
    norm_df, stats = process_data(csv_path)
    save_processed_data(norm_df, stats, processed_data_path)

    # Generate and save predictions
    predict_and_save_all_players(norm_df, current_season, stats, predictions_path)

    print("Pre-processing complete. Data and predictions saved.")

if __name__ == "__main__":
    main()
