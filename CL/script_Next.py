import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import xgboost as xgb
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
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf = xgb.XGBClassifier(random_state=42)

clf.fit(X_train, y_train)

# Step 5: Use the model to predict outcomes in nextGames.csv
next_games = pd.read_csv('nextGames.csv')
next_games['Heim'] = next_games['Heim'].astype('category')
next_games['Auswärts'] = next_games['Auswärts'].astype('category')

# Create dummy variables using the same columns as X_train
X_pred = pd.get_dummies(next_games[['Heim', 'Auswärts']]).reindex(columns=X_train.columns, fill_value=0)
y_pred = clf.predict(X_pred)
next_games[['Heimgewinn', 'Auswärtsgewinn', 'Unentschieden']] = y_pred

# Create a new column 'Punkte Heim' and assign points based on predictions
next_games['Punkte Heim'] = np.where(next_games['Heimgewinn'] > next_games['Auswärtsgewinn'], 3,
                                     np.where(next_games['Heimgewinn'] == next_games['Auswärtsgewinn'], 1, 0))

# Create a new column 'Punkte Auswärts' and assign points based on predictions
next_games['Punkte Auswärts'] = np.where(next_games['Auswärtsgewinn'] > next_games['Heimgewinn'], 3,
                                        np.where(next_games['Heimgewinn'] == next_games['Auswärtsgewinn'], 1, 0))

# Drop the prediction columns and save the results to a CSV file
next_games.drop(['Heimgewinn', 'Auswärtsgewinn', 'Unentschieden', 'Wo', 'Tag', 'Uhrzeit', 'Ergebnis',
                 'Zuschauerzahl', 'Spielort', 'Schiedsrichter', 'Spielbericht'], axis=1, inplace=True)

next_games.to_csv('predictions.csv', index=False)
