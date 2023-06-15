import requests
import pandas as pd
import math

# API key
api_key = (open('api_key.txt', mode='r')).read()

# Base URL for API requests
base_url = 'https://v2.api-football.com/'

# Function to get API data
def get_api_data(end_url):
    url = base_url + end_url
    headers = {'X-RapidAPI-Key': api_key}
    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        raise RuntimeError(f'error {res.status_code}')
    return res.text

# Function to slice API data
def slice_api(api_str_output, start_char, end_char):
    return api_str_output[start_char:-end_char]

# Function to save API output to a file
def save_api_output(save_name, json_data):
    with open(save_name + '.json', 'w') as writeFile:
        writeFile.write(json_data)

# Function to read JSON data as a pandas DataFrame
def read_json_as_pd_df(json_data, orient_def='records'):
    return pd.read_json(json_data, orient=orient_def)

# Requesting Swiss Super League fixtures by season
def req_ssl_fixtures_id(season_code):
    ssl_fixtures_raw = get_api_data(f'/fixtures/league/{season_code}/')
    ssl_fixtures_sliced = slice_api(ssl_fixtures_raw, 33, 2)
    save_api_output(f'ssl_fixtures', ssl_fixtures_sliced)
    return read_json_as_pd_df('ssl_fixtures.json')

# Requesting Swiss Super League fixtures for the specified year
YEAR = 2023
if YEAR == 2022:
    season_id = 4389
elif YEAR == 2023:
    season_id = 4956
else:
    print('Please lookup season id and specify it as the season_id variable')

fixtures = req_ssl_fixtures_id(season_id)

# Clean fixtures list
fixtures_clean = pd.DataFrame({
    'Fixture ID': fixtures['fixture_id'],
    'Game Date': fixtures['event_date'].str[:10],
    'Home Team ID': fixtures['homeTeam'].str[12:14].astype(int),
    'Away Team ID': fixtures['awayTeam'].str[12:14].astype(int),
    'Home Team Goals': fixtures['goalsHomeTeam'],
    'Away Team Goals': fixtures['goalsAwayTeam'],
    'Venue': fixtures['venue'],
    'Home Team': fixtures['homeTeam'].apply(lambda x: x['team_name']),
    'Away Team': fixtures['awayTeam'].apply(lambda x: x['team_name']),
    'Home Team Logo': fixtures['homeTeam'].apply(lambda x: x['logo']),
    'Away Team Logo': fixtures['awayTeam'].apply(lambda x: x['logo'])
})

fixtures_clean.to_csv(f'{YEAR}_ssl_fixtures_df.csv', index=False)

print('SSL fixtures for the year', YEAR, 'have been fetched and saved.')
