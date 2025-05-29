import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
import glob

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
    # Better path handling for both local and GitHub Actions environments
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    
    # Define possible file locations
    possible_locations = [
        "../avocado_ripeness_dataset.csv",                # Relative from script directory
        os.path.join(repo_root, "avocado_ripeness_dataset.csv"),  # Root directory
        "avocado_ripeness_dataset.csv",                   # Current directory
        os.path.join(repo_root, "data", "avocado_ripeness_dataset.csv")  # Data directory if exists
    ]
    
    # Find the first location that exists
    input_file = None
    for loc in possible_locations:
        if os.path.exists(loc):
            input_file = loc
            print(f"Found dataset at: {input_file}")
            break
    
    # If not found in predefined locations, search the repository
    if input_file is None:
        print("Searching for dataset file...")
        matches = []
        for root, _, files in os.walk(repo_root):
            for file in files:
                if file == "avocado_ripeness_dataset.csv":
                    matches.append(os.path.join(root, file))
        
        if matches:
            input_file = matches[0]
            print(f"Found dataset at: {input_file}")
        else:
            print("ERROR: Could not find avocado_ripeness_dataset.csv anywhere")
            print(f"Current directory: {os.getcwd()}")
            print(f"Repository contents:")
            for item in glob.glob(os.path.join(repo_root, '*')):
                print(f"  {item}")
            sys.exit(1)
    
    df = preprocess_data(input_file)
    
    output_path = os.path.join(script_dir, "avocado_ripeness_dataset_preprocessed.csv")
    df.to_csv(output_path, index=False)
    print(f"Preprocessing complete! Output saved to {output_path}")