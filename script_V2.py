import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.model_selection import train_test_split
import numpy as np
import csv


def preprocess_data(data):
    data['Ergebnis'] = data['Ergebnis'].apply(lambda x: [int(i) for i in x.split('–')])
    data['Heimgewinn'] = data['Ergebnis'].apply(lambda x: int(x[0] > x[1]))
    data['Auswärtsgewinn'] = data['Ergebnis'].apply(lambda x: int(x[0] < x[1]))
    data['Unentschieden'] = data['Ergebnis'].apply(lambda x: int(x[0] == x[1]))
    data['Heim'] = data['Heim'].str.replace('Zürich', 'Zurich')
    data['Auswärts'] = data['Auswärts'].str.replace('Zürich', 'Zurich')
    data['Heim'] = data['Heim'].astype('category')
    data['Auswärts'] = data['Auswärts'].astype('category')
    return data


def train_model(X_train, y_train):
    # clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf = xgb.XGBClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf


def predict_outcomes(model, data):
    X_pred = pd.get_dummies(data[['Heim', 'Auswärts']])
    y_pred = model.predict(X_pred)
    data[['Heimgewinn', 'Auswärtsgewinn', 'Unentschieden']] = y_pred
    return data


def calculate_points(data):
    points = {}
    for _, row in data.iterrows():
        home_team, away_team = row['Heim'], row['Auswärts']
        home_points, away_points = int(row['Punkte Heim']), int(row['Punkte Auswärts'])
        if home_team not in points:
            points[home_team] = 0
        if away_team not in points:
            points[away_team] = 0
        points[home_team] += home_points
        points[away_team] += away_points
    return points


def write_points_to_csv(points, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Team', 'Points'])
        for team, team_points in points.items():
            writer.writerow([team, team_points])


def update_table(points_file, table_file, new_table_file):
    points_reader = csv.DictReader(points_file)
    table_reader = csv.DictReader(table_file)
    table_writer = csv.DictWriter(new_table_file, fieldnames=table_reader.fieldnames)

    points_dict = {row['Team']: int(row['Points']) for row in points_reader}

    new_rows = []
    for row in table_reader:
        team = row['Verein']
        points = points_dict.get(team, 0)
        row['Pkt'] = str(int(row['Pkt']) + points)
        new_rows.append(row)

    sorted_rows = sorted(new_rows, key=lambda x: int(x['Pkt']), reverse=True)

    table_writer.writeheader()
    for row in sorted_rows:
        table_writer.writerow(row)

def main():
    # Step 1: Read the data from data.csv
    df = pd.read_csv('SL/data.csv')

    # Step 2: Preprocess the data
    df = preprocess_data(df)

    # Step 3: Split the data into training and testing sets
    X = pd.get_dummies(df[['Heim', 'Auswärts']])
    y = df[['Heimgewinn', 'Auswärtsgewinn', 'Unentschieden']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Train a machine learning model
    model = train_model(X_train, y_train)

    # Step 5: Use the model to predict outcomes in lastGames.csv
    last_games = pd.read_csv('SL/lastGames.csv')
    last_games['Heim'] = last_games['Heim'].str.replace('Zürich', 'Zurich')
    last_games['Auswärts'] = last_games['Auswärts'].str.replace('Zürich', 'Zurich')

    last_games['Heim'] = last_games['Heim'].astype('category')
    last_games['Auswärts'] = last_games['Auswärts'].astype('category')
    last_games = predict_outcomes(model, last_games)

    # Create a new column 'Punkte Heim' and assign points based on predictions
    last_games['Punkte Heim'] = np.where(last_games['Heimgewinn'] > last_games['Auswärtsgewinn'], 3,
                                         np.where(last_games['Heimgewinn'] == last_games['Auswärtsgewinn'], 1, 0))

    # Create a new column 'Punkte Auswärts' and assign points based on predictions
    last_games['Punkte Auswärts'] = np.where(last_games['Auswärtsgewinn'] > last_games['Heimgewinn'], 3,
                                            np.where(last_games['Heimgewinn'] == last_games['Auswärtsgewinn'], 1, 0))

    # Drop the prediction columns and save the results to a CSV file
    last_games.drop(['Heimgewinn', 'Auswärtsgewinn', 'Unentschieden', 'Wo', 'Tag', 'Uhrzeit', 'Ergebnis',
                     'Zuschauerzahl', 'Spielort', 'Schiedsrichter', 'Spielbericht', 'Hinweise'], axis=1, inplace=True)
    last_games.to_csv('SL/predictions.csv', index=False)

    # Calculate points for each team
    points = calculate_points(last_games)

    # Write points to a CSV file
    write_points_to_csv(points, 'SL/points.csv')

    # Open the files
    with open('SL/points.csv', 'r') as points_file, open('SL/tabelle.csv', 'r') as table_file, \
            open('SL/new_tabelle.csv', 'w', newline='') as new_table_file:

        # Update the table with points
        update_table(points_file, table_file, new_table_file)

    # Close the files
    points_file.close()
    table_file.close()
    new_table_file.close()


if __name__ == '__main__':
    main()