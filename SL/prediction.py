import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost as xgb

# Load the result data from 2020-2022 CSV files
df_2020 = pd.read_csv('data_2020.csv')
df_2021 = pd.read_csv('data_2021.csv')
df_2022 = pd.read_csv('data_2022.csv')

# Combine the datasets
df = pd.concat([df_2020, df_2021, df_2022], ignore_index=True)

# Step 3: Split the data into training and testing sets
X = pd.get_dummies(df[['Heim', 'Auswärts']])
y = df[['Heimgewinn', 'Auswärtsgewinn', 'Unentschieden']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a machine learning model
clf = xgb.XGBClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 5: Use the model to predict outcomes in the 2023 CSV file
next_games = pd.read_csv('data_2023.csv')
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
next_games.drop(['Ergebnis', 'Heimgewinn', 'Auswärtsgewinn', 'Unentschieden'], axis=1, inplace=True)
next_games.to_csv('predictions_2023.csv', index=False)

# Load the predictions CSV file for 2023 matches
predictions = pd.read_csv('predictions_2023.csv')

# Calculate average points per game
teams = predictions['Heim'].unique()
avg_points_per_game = []
for team in teams:
    team_predictions = predictions.loc[(predictions['Heim'] == team) | (predictions['Auswärts'] == team)]
    points_per_game = 0
    total_games = 0
    for _, row in team_predictions.iterrows():
        win_prob = row['Heimgewinn Chance']
        draw_prob = row['Unentschieden Chance']
        loss_prob = row['Auswärtsgewinn Chance']
        points_per_game += (3 * win_prob + 1 * draw_prob) / (win_prob + draw_prob + loss_prob)
        total_games += 1
    avg_points_per_game.append(points_per_game / total_games)  # Divide by total games played

# Create probability table
prob_table = pd.DataFrame({'Team': teams, 'Average Points per Game': avg_points_per_game})

# Sort the table by average points per game in descending order
prob_table = prob_table.sort_values(by='Average Points per Game', ascending=False)

# Reset the index
prob_table.reset_index(drop=True, inplace=True)

# Print the probability table
print(prob_table)

