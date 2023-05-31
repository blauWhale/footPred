import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
next_games = pd.read_csv('final.csv')
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

# Calculate and print ML metrics
y_true = y_test.values
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
