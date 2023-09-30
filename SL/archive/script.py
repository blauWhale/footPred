import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import csv


# Step 1: Read the data from data.csv
df = pd.read_csv('data.csv')

# Step 2: Preprocess the data
df['Ergebnis'] = df['Ergebnis'].apply(lambda x: [int(i) for i in x.split('–')])
df['Heimgewinn'] = df['Ergebnis'].apply(lambda x: int(x[0] > x[1]))
df['Auswärtsgewinn'] = df['Ergebnis'].apply(lambda x: int(x[0] < x[1]))
df['Unentschieden'] = df['Ergebnis'].apply(lambda x: int(x[0] == x[1]))
df['Heim'] = df['Heim'].astype('category')
df['Auswärts'] = df['Auswärts'].astype('category')

# Step 3: Split the data into training and testing sets
X = pd.get_dummies(df[['Heim', 'Auswärts']])
y = df[['Heimgewinn', 'Auswärtsgewinn', 'Unentschieden']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a machine learning model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 5: Use the model to predict outcomes in lastGames.csv
last_games = pd.read_csv('lastGames.csv')
last_games['Heim'] = last_games['Heim'].astype('category')
last_games['Auswärts'] = last_games['Auswärts'].astype('category')
X_pred = pd.get_dummies(last_games[['Heim', 'Auswärts']])
y_pred = clf.predict(X_pred)
last_games[['Heimgewinn', 'Auswärtsgewinn', 'Unentschieden']] = y_pred

# Create a new column 'Punkte Heim' and assign points based on predictions
last_games['Punkte Heim'] = np.where(last_games['Heimgewinn'] > last_games['Auswärtsgewinn'], 3, np.where(last_games['Heimgewinn'] == last_games['Auswärtsgewinn'], 1, 0))

# Create a new column 'Punkte Auswärts' and assign points based on predictions
last_games['Punkte Auswärts'] = np.where(last_games['Auswärtsgewinn'] > last_games['Heimgewinn'], 3, np.where(last_games['Heimgewinn'] == last_games['Auswärtsgewinn'], 1, 0))

# Drop the prediction columns and save the results to a CSV file
last_games.drop(['Heimgewinn', 'Auswärtsgewinn', 'Unentschieden', 'Wo', 'Tag', 'Uhrzeit', 'Ergebnis', 'Zuschauerzahl', 'Spielort', 'Schiedsrichter', 'Spielbericht', 'Hinweise'], axis=1, inplace=True)
last_games.to_csv('predictions.csv', index=False)



points = {}

with open('predictions.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader) # Skip header row
    for row in reader:
        home_team, away_team = row[1], row[2]
        home_points, away_points = int(row[3]), int(row[4])
        # Update points for home team
        if home_team not in points:
            points[home_team] = 0
        points[home_team] += home_points
        # Update points for away team
        if away_team not in points:
            points[away_team] = 0
        points[away_team] += away_points

with open('points.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Team', 'Points'])
    for team, team_points in points.items():
        writer.writerow([team, team_points])

# Open the files
points_file = open('points.csv', 'r')
table_file = open('tabelle.csv', 'r')
new_table_file = open('new_tabelle.csv', 'w', newline='')

# Create CSV reader and writer objects
points_reader = csv.DictReader(points_file)
table_reader = csv.DictReader(table_file)
table_writer = csv.DictWriter(new_table_file, fieldnames=table_reader.fieldnames)

# Create a dictionary to map teams to points
points_dict = {row['Team']: int(row['Points']) for row in points_reader}

# Iterate through the table and add points to each team
new_rows = []
for row in table_reader:
    team = row['Verein']
    points = points_dict.get(team, 0)
    row['Pkt'] = str(int(row['Pkt']) + points)
    new_rows.append(row)

# Sort the rows by points
sorted_rows = sorted(new_rows, key=lambda x: int(x['Pkt']), reverse=True)

# Write the sorted rows to the new CSV file
table_writer.writeheader()
for row in sorted_rows:
    table_writer.writerow(row)

# Close the files
points_file.close()
table_file.close()
new_table_file.close()