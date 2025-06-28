import pandas as pd
import os

# Load the original dataset to capture column structure
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "Housing.csv"))

# Encode binary features
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[binary_cols] = df[binary_cols].applymap(lambda x: 1 if x == 'yes' else 0)

# One-hot encode 'furnishingstatus'
df = pd.get_dummies(df, columns=['furnishingstatus'])

# Save the order of columns used for training
feature_columns = df.drop(columns=["price"]).columns.tolist()

def preprocess_input(data: dict):
    # Convert input dict to DataFrame
    input_df = pd.DataFrame([data])
    
    # Binary encode input
    input_df[binary_cols] = input_df[binary_cols].applymap(lambda x: 1 if x == 'yes' else 0)

    # One-hot encode 'furnishingstatus'
    input_df = pd.get_dummies(input_df, columns=['furnishingstatus'])

    # Add missing furnishingstatus columns (if any)
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder to match training data
    input_df = input_df[feature_columns]

    return input_df
