import pandas as pd
import requests

def get_fixtures_data(league, season):
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
    querystring = {"league": league, "season": season}
    headers = {
        "X-RapidAPI-Key": "df439bb8dfmsh328a99e24f9725ep1c783ajsn5976979dfa31",
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()
    fixtures = data["response"]
    fixture_list = []
    for fixture in fixtures:
        home_goals = fixture['goals']['home']
        away_goals = fixture['goals']['away']
        if home_goals is None or away_goals is None:
            result = None
        else:
            result = f"{home_goals}–{away_goals}"
        fixture_dict = {
            "Heim": fixture["teams"]["home"]["name"],
            "Auswärts": fixture["teams"]["away"]["name"],
            "Ergebnis": result,
            "Datum": pd.to_datetime(fixture.get("fixture", {}).get("date")).strftime('%Y-%m-%d %H:%M:%S')
        }
        fixture_list.append(fixture_dict)
    return fixture_list

# Get data for 2023 season
fixture_list_2023 = get_fixtures_data("207", "2023")
df_2023 = pd.DataFrame(fixture_list_2023)
df_2023['Ergebnis'] = df_2023['Ergebnis'].apply(lambda x: [int(i) for i in x.split('–')] if x is not None else None)
df_2023['Heimgewinn'] = df_2023['Ergebnis'].apply(lambda x: int(x[0] > x[1]) if x is not None else None)
df_2023['Auswärtsgewinn'] = df_2023['Ergebnis'].apply(lambda x: int(x[0] < x[1]) if x is not None else None)
df_2023['Unentschieden'] = df_2023['Ergebnis'].apply(lambda x: int(x[0] == x[1]) if x is not None else None)
df_2023['Heim'] = df_2023['Heim'].astype('category')
df_2023['Auswärts'] = df_2023['Auswärts'].astype('category')
df_2023.to_csv('data_2023.csv', index=False)

# Get data for 2022 season
fixture_list_2022 = get_fixtures_data("207", "2022")
df_2022 = pd.DataFrame(fixture_list_2022)
df_2022['Ergebnis'] = df_2022['Ergebnis'].apply(lambda x: [int(i) for i in x.split('–')] if x is not None else None)
df_2022['Heimgewinn'] = df_2022['Ergebnis'].apply(lambda x: int(x[0] > x[1]) if x is not None else None)
df_2022['Auswärtsgewinn'] = df_2022['Ergebnis'].apply(lambda x: int(x[0] < x[1]) if x is not None else None)
df_2022['Unentschieden'] = df_2022['Ergebnis'].apply(lambda x: int(x[0] == x[1]) if x is not None else None)
df_2022['Heim'] = df_2022['Heim'].astype('category')
df_2022['Auswärts'] = df_2022['Auswärts'].astype('category')
df_2022.to_csv('data_2022.csv', index=False)

# Get data for 2021 season
fixture_list_2021 = get_fixtures_data("207", "2021")
df_2021 = pd.DataFrame(fixture_list_2021)
df_2021['Ergebnis'] = df_2021['Ergebnis'].apply(lambda x: [int(i) for i in x.split('–')] if x is not None else None)
df_2021['Heimgewinn'] = df_2021['Ergebnis'].apply(lambda x: int(x[0] > x[1]) if x is not None else None)
df_2021['Auswärtsgewinn'] = df_2021['Ergebnis'].apply(lambda x: int(x[0] < x[1]) if x is not None else None)
df_2021['Unentschieden'] = df_2021['Ergebnis'].apply(lambda x: int(x[0] == x[1]) if x is not None else None)
df_2021['Heim'] = df_2021['Heim'].astype('category')
df_2021['Auswärts'] = df_2021['Auswärts'].astype('category')
df_2021.to_csv('data_2021.csv', index=False)

# Get data for 2020 season
fixture_list_2020 = get_fixtures_data("207", "2020")
df_2020 = pd.DataFrame(fixture_list_2020)
df_2020['Ergebnis'] = df_2020['Ergebnis'].apply(lambda x: [int(i) for i in x.split('–')] if x is not None else None)
df_2020['Heimgewinn'] = df_2020['Ergebnis'].apply(lambda x: int(x[0] > x[1]) if x is not None else None)
df_2020['Auswärtsgewinn'] = df_2020['Ergebnis'].apply(lambda x: int(x[0] < x[1]) if x is not None else None)
df_2020['Unentschieden'] = df_2020['Ergebnis'].apply(lambda x: int(x[0] == x[1]) if x is not None else None)
df_2020['Heim'] = df_2020['Heim'].astype('category')
df_2020['Auswärts'] = df_2020['Auswärts'].astype('category')
df_2020.to_csv('data_2020.csv', index=False)
