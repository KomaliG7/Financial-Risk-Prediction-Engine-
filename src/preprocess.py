import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def main(input_path, output_dir):
    # Load data
    print(f"Loading dataset from {input_path}...")
    data = pd.read_csv(input_path)
    
    # Check if 'Class' column exists
    if 'Class' not in data.columns:
        raise ValueError("Dataset must contain a 'Class' column as target variable.")

    # Split features and target
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save as .npz
    np.savez_compressed(os.path.join(output_dir, 'train.npz'), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(output_dir, 'test.npz'), X=X_test, y=y_test)

    print(f"Preprocessed data saved to {output_dir}/train.npz and test.npz")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw data for training.")
    parser.add_argument("--input", required=True, help="Path to raw CSV file")
    parser.add_argument("--output", required=True, help="Directory to save processed files")
    args = parser.parse_args()

    main(args.input, args.output)
