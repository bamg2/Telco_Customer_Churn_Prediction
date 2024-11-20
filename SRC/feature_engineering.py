from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def encode_features(df):
    """Encodes categorical features using one-hot encoding."""
    categorical_cols = df.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
    
    df.drop(columns=categorical_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, encoded_df], axis=1)
    
    return df
