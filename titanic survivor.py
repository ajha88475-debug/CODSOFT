import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load the Dataset
# Ensure the file 'Titanic-Dataset.csv' is in your folder
try:
    data = pd.read_csv('Titanic-Dataset.csv')
except FileNotFoundError:
    raise FileNotFoundError("Titanic-Dataset.csv not found in the current folder. Place the CSV next to this script.")

# 2. Data Exploration & Cleaning
print(data.head())
print(data.info())

# Drop columns that likely won't help prediction (Name, Ticket, Cabin, PassengerId)
data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Handle missing values: Fill missing Age with the mean age
data['Age'].fillna(data['Age'].mean(), inplace=True)

# Fill missing Embarked values with the mode (most common port)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Fill missing Fare values (common source of errors)
if 'Fare' in data.columns:
    data['Fare'].fillna(data['Fare'].median(), inplace=True)

# 3. Encoding Categorical Variables
# Convert 'Sex' to numerical: Male=0, Female=1
data.replace({'Sex':{'male':0, 'female':1}, 'Embarked':{'S':0, 'C':1, 'Q':2}}, inplace=True)

# Ensure all feature columns are numeric (coerce any lingering non-numeric)
data = data.apply(pd.to_numeric, errors='coerce')

# If coercion introduced NaNs, fill them (fallback)
data.fillna(data.median(numeric_only=True), inplace=True)

# 4. Splitting Features and Target
X = data.drop(columns=['Survived'], axis=1) # Features (Age, Sex, Pclass, etc.)
Y = data['Survived']                        # Target (0 = Dead, 1 = Survived)

# Split into training and test data (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# 5. Model Training
# increase max_iter to avoid convergence errors
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# 6. Evaluation
# Prediction on training data
train_prediction = model.predict(X_train)
print(f"Accuracy on Training data: {accuracy_score(Y_train, train_prediction)}")

# Prediction on test data
test_prediction = model.predict(X_test)
print(f"Accuracy on Test data: {accuracy_score(Y_test, test_prediction)}")