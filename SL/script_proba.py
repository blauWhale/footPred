import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost as xgb

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

# Assign chance percentages to the columns
next_games[['Heimgewinn Chance', 'Auswärtsgewinn Chance', 'Unentschieden Chance']] = y_pred_proba

# Drop unnecessary columns and save the results to a CSV file
next_games.drop(['Wo', 'Tag', 'Uhrzeit', 'Ergebnis', 'Zuschauerzahl', 'Spielort', 'Schiedsrichter', 'Spielbericht', 'Hinweise'], axis=1, inplace=True)
next_games.to_csv('predictions.csv', index=False)
