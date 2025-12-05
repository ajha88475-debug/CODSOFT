import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore")

# --- 1. Load Data ---
# Use the path that worked for you before
try:
    data = pd.read_csv("IMDb Movies India.csv", encoding='latin1')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: File not found. Make sure the path is correct.")
    exit()

# --- 2. Data Cleaning (THE FIX IS HERE) ---
# We must drop rows with missing values in 'Duration', 'Year', or 'Votes' BEFORE modifying them
data.dropna(subset=['Year', 'Duration', 'Votes', 'Rating', 'Actor 1', 'Actor 2', 'Actor 3', 'Director', 'Genre'], inplace=True)

# Clean 'Year': Remove parenthesis like '(2019)' -> 2019
data['Year'] = data['Year'].str.replace(r'[()]', '', regex=True).astype(int)

# Clean 'Duration': Remove ' min' like '120 min' -> 120
data['Duration'] = data['Duration'].str.replace(' min', '').astype(int)

# Clean 'Votes': Remove commas like '1,000' -> 1000
data['Votes'] = data['Votes'].str.replace(',', '').astype(int)

# --- 3. Feature Engineering (Text to Numbers) ---
def mean_encode(df, col, target='Rating'):
    mean_ratings = df.groupby(col)[target].mean()
    df[col] = df[col].map(mean_ratings)
    return df

# Encode the categorical columns
for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    data = mean_encode(data, col)

# --- 4. Split Data ---
X = data[['Year', 'Duration', 'Votes', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']]
Y = data['Rating']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# --- 5. Train Model ---
print("Training the Regression Model...")
model = LinearRegression()
model.fit(X_train, Y_train)

# --- 6. Evaluate ---
y_pred = model.predict(X_test)
score = r2_score(Y_test, y_pred)

print("\n")
print("------------------------------------------------")
print(f"Model R^2 Score: {score:.4f}")
print("------------------------------------------------")
print("\n")