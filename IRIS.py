import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

# Ignore warnings for clean output
warnings.filterwarnings("ignore")

# --- 1. Load Data ---
# We try to load from the Desktop. Update the path if your file is somewhere else.
try:
    data = pd.read_csv("IRIS.csv")
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'IRIS.csv' not found. Please place it on your Desktop.")
    exit()

# --- 2. Data Inspection ---
print("First 5 rows:")
print(data.head())

# --- 3. Preprocessing ---
# The 'species' column contains text (e.g., 'Iris-setosa').
# Machines understand numbers better.
# We map: Iris-setosa -> 0, Iris-versicolor -> 1, Iris-virginica -> 2
# Note: Some datasets use 'Species' (capital S) or 'species' (lowercase).
# We check the column name first.
if 'species' in data.columns:
    target_col = 'species'
elif 'Species' in data.columns:
    target_col = 'Species'
else:
    print("Error: Could not find a 'species' column.")
    exit()

# Map the text to numbers
mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
data[target_col] = data[target_col].map(mapping)

# --- 4. Split Data ---
# Features: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
# (Note: Your CSV might have slightly different column names like 'sepal_length' or 'SepalLengthCm')
# We drop the 'Id' column if it exists because it's just a row number.
if 'Id' in data.columns:
    data = data.drop(columns=['Id'])

X = data.drop(columns=[target_col])
Y = data[target_col]

# Split 80% training, 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# --- 5. Train Model ---
print("Training the model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# --- 6. Evaluate ---
prediction = model.predict(X_test)
accuracy = accuracy_score(Y_test, prediction)

print("\n")
print("------------------------------------------------")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("------------------------------------------------")
print("\n")

# --- 7. Test with New Data ---
# Let's predict a random flower: Sepal Length=5.1, Sepal Width=3.5, Petal Length=1.4, Petal Width=0.2
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])

# Warning: The model expects a dataframe structure, but numpy array works too (ignoring warning)
pred = model.predict(new_flower)

# Convert number back to name
reverse_mapping = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
print(f"Prediction for new flower {new_flower}: {reverse_mapping[pred[0]]}")