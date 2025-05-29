import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    # standarisasi
    numerical_column = df.select_dtypes(include='number').columns.tolist()
    scaler = StandardScaler()
    df[numerical_column] = scaler.fit_transform(df[numerical_column])
    
    # one-hot encoding
    categorical_column = ['color_category']
    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform(df[categorical_column])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_column))
    df = pd.concat([df, encoded_df], axis=1)
    df.drop(columns=categorical_column, inplace=True)
    
    return df

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else "../avocado_ripeness_dataset.csv"
    df = preprocess_data(input_file)
    df.to_csv("avocado_ripeness_dataset_preprocessed.csv", index=False)