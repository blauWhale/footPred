import requests
import json
import csv

def get_fixtures_for_round(league, season, round_number):
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
    querystring = {"league": league, "season": season, "round": round_number}
    headers = {
        "X-RapidAPI-Key": "df439bb8dfmsh328a99e24f9725ep1c783ajsn5976979dfa31",
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()

    fixtures = data.get("response", [])
    if not fixtures:
        print(f"No fixtures found for league {league}, season {season}, and round {round_number}.")
        return []

    fixture_ids = [str(fixture.get("fixture", {}).get("id", "")) for fixture in fixtures]
    return fixture_ids

def get_and_save_predictions_for_round(league, season, round_number):
    fixture_ids = get_fixtures_for_round(league, season, round_number)

    if not fixture_ids:
        print("No fixtures to process.")
        return

    # Prepare data for CSV
    csv_data = []
    headers = ["Fixture ID", "Date", "Home Team", "Away Team", "Winner", "Advice", "Home Win %", "Draw %", "Away Win %",
               "Home Form %", "Away Form %", "Home Att %", "Away Att %", "Home Def %", "Away Def %",
               "Home Poisson %", "Away Poisson %", "Home H2H %", "Away H2H %", "Home Goals %", "Away Goals %",
               "Home Total %", "Away Total %"]
    csv_data.append(headers)

    for fixture_id in fixture_ids:
        get_prediction_data(fixture_id, csv_data)

    # Save data to CSV file
    csv_filename = "predictions_data.csv"
    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(csv_data)

    print(f"Data saved to {csv_filename}.")


def get_prediction_data(fixture, csv_data):
    url = "https://api-football-v1.p.rapidapi.com/v3/predictions"
    querystring = {"fixture": fixture}
    headers = {
        "X-RapidAPI-Key": "df439bb8dfmsh328a99e24f9725ep1c783ajsn5976979dfa31",
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()

    # Extract relevant information
    predictions = data.get("response", [])
    if not predictions:
        print("No predictions found in the JSON data for fixture ID:", fixture)
        return

    for prediction in predictions:
        fixture_id = fixture
        date = prediction.get("fixture", {}).get("date", "")
        home_team = prediction.get("teams", {}).get("home", {}).get("name", "")
        away_team = prediction.get("teams", {}).get("away", {}).get("name", "")
        winner = prediction.get("predictions", {}).get("winner", {}).get("name", "")
        advice = prediction.get("predictions", {}).get("advice", "")
        home_win_percent = prediction.get("predictions", {}).get("percent", {}).get("home", "")
        draw_percent = prediction.get("predictions", {}).get("percent", {}).get("draw", "")
        away_win_percent = prediction.get("predictions", {}).get("percent", {}).get("away", "")

        # Added comparison percentages
        home_form_percent = prediction.get("comparison", {}).get("form", {}).get("home", "")
        away_form_percent = prediction.get("comparison", {}).get("form", {}).get("away", "")
        home_att_percent = prediction.get("comparison", {}).get("att", {}).get("home", "")
        away_att_percent = prediction.get("comparison", {}).get("att", {}).get("away", "")
        home_def_percent = prediction.get("comparison", {}).get("def", {}).get("home", "")
        away_def_percent = prediction.get("comparison", {}).get("def", {}).get("away", "")
        home_poisson_percent = prediction.get("comparison", {}).get("poisson_distribution", {}).get("home", "")
        away_poisson_percent = prediction.get("comparison", {}).get("poisson_distribution", {}).get("away", "")
        home_h2h_percent = prediction.get("comparison", {}).get("h2h", {}).get("home", "")
        away_h2h_percent = prediction.get("comparison", {}).get("h2h", {}).get("away", "")
        home_goals_percent = prediction.get("comparison", {}).get("goals", {}).get("home", "")
        away_goals_percent = prediction.get("comparison", {}).get("goals", {}).get("away", "")
        home_total_percent = prediction.get("comparison", {}).get("total", {}).get("home", "")
        away_total_percent = prediction.get("comparison", {}).get("total", {}).get("away", "")

        row = [fixture_id, date, home_team, away_team, winner, advice, home_win_percent, draw_percent, away_win_percent,
               home_form_percent, away_form_percent, home_att_percent, away_att_percent, home_def_percent,
               away_def_percent, home_poisson_percent, away_poisson_percent, home_h2h_percent, away_h2h_percent,
               home_goals_percent, away_goals_percent, home_total_percent, away_total_percent]

        csv_data.append(row)



# Example usage
league_id = "207"
season_year = "2023"
round_number = "Regular Season - 28"
get_and_save_predictions_for_round(league_id, season_year, round_number)
