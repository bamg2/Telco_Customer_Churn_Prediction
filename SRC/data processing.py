import pandas as pd

def load_data(filepath):
    """Loads the dataset."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Performs data preprocessing steps."""
    # Convert TotalCharges to numeric and handle missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    # Drop unnecessary columns
    df.drop('customerID', axis=1, inplace=True)

    return df
