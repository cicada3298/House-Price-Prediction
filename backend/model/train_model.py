import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
import os

# Load the dataset
csv_path = os.path.join(os.path.dirname(__file__), "Housing.csv")
df = pd.read_csv(csv_path)

# Binary encode yes/no columns
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[binary_cols] = df[binary_cols].applymap(lambda x: 1 if x == 'yes' else 0)

# One-hot encode 'furnishingstatus'
df = pd.get_dummies(df, columns=['furnishingstatus'])

# Split into features and target
X = df.drop(columns=["price"])  # ✅ all features except target
y = df["price"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
joblib.dump(model, os.path.join(os.path.dirname(__file__), "model.pkl"))
print("✅ Model trained and saved.")
