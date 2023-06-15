import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost as xgb
import requests


# Make the API call to fetch the fixtures data
url = "https://api-football.com/v3/fixtures"
params = {
    "league": "207",  # Premier League
    "season": "2022"
    # Add any other parameters you may need
}
headers = {
	"X-RapidAPI-Key": "df439bb8dfmsh328a99e24f9725ep1c783ajsn5976979dfa31",
	"X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
}
response = requests.get(url, params=params, headers=headers)
data = response.json()

# Extract the fixture data and preprocess it
fixtures = data["response"]
fixture_list = []
for fixture in fixtures:
    fixture_dict = {
        "Heim": fixture["teams"]["home"]["name"],
        "Auswärts": fixture["teams"]["away"]["name"],
        "Ergebnis": f"{fixture['goals']['home']}–{fixture['goals']['away']}",
        "Schiedsrichter": fixture["referee"],
        "Datum": fixture["date"]
    }
    fixture_list.append(fixture_dict)

df = pd.DataFrame(fixture_list)

# Preprocess the data
df['Ergebnis'] = df['Ergebnis'].apply(lambda x: [int(i) for i in x.split('–')])
df['Heimgewinn'] = df['Ergebnis'].apply(lambda x: int(x[0] > x[1]))
df['Auswärtsgewinn'] = df['Ergebnis'].apply(lambda x: int(x[0] < x[1]))
df['Unentschieden'] = df['Ergebnis'].apply(lambda x: int(x[0] == x[1]))
df['Heim'] = df['Heim'].astype('category')
df['Auswärts'] = df['Auswärts'].astype('category')

# Save the data to the data.csv file
df.to_csv('data.csv', index=False)



# Step 3: Split the data into training and testing sets
X = pd.get_dummies(df[['Heim', 'Auswärts']])
y = df[['Heimgewinn', 'Auswärtsgewinn', 'Unentschieden']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a machine learning model
clf = xgb.XGBClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 5: Use the model to predict outcomes in nextGames.csv
next_games = pd.read_csv('nextGames.csv')
next_games['Heim'] = next_games['Heim'].astype('category')
next_games['Auswärts'] = next_games['Auswärts'].astype('category')

# Create dummy variables using the same columns as X_train
X_pred = pd.get_dummies(next_games[['Heim', 'Auswärts']]).reindex(columns=X_train.columns, fill_value=0)

# Predict the probabilities for each class label
y_pred_proba = clf.predict_proba(X_pred)

# Normalize the probabilities
y_pred_proba_normalized = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)

# Assign chance percentages to the columns
next_games[['Heimgewinn Chance', 'Auswärtsgewinn Chance', 'Unentschieden Chance']] = y_pred_proba_normalized

# Drop unnecessary columns and save the results to a CSV file
next_games.drop(['Wo', 'Tag', 'Uhrzeit', 'Ergebnis', 'Zuschauerzahl', 'Spielort', 'Schiedsrichter', 'Spielbericht', 'Hinweise'], axis=1, inplace=True)
next_games.to_csv('predictions.csv', index=False)

# Read the predictions CSV file
predictions = pd.read_csv('predictions.csv')

# Drop the 'Datum' column
predictions.drop('Datum', axis=1, inplace=True)

# Combine 'Heim' and 'Auswärts' into 'Spiel' column
predictions['Spiel'] = predictions['Heim'] + ' vs ' + predictions['Auswärts']

# Reorder the columns
predictions = predictions[['Spiel', 'Heimgewinn Chance','Unentschieden Chance','Auswärtsgewinn Chance']]

# Save the formatted predictions to a new CSV file
predictions.to_csv('prediction_formatted.csv', index=False)

