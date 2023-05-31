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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)

# Step 4: Train a machine learning model
clf = xgb.XGBClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 5: Evaluate the model on the test set and print metrics
y_true = y_test.values
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
