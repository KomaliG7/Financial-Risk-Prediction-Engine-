# src/data_prep.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

def load_data(path):
    df = pd.read_csv(path)
    return df

def build_preprocessor(df, categorical_threshold=20):
    # Identify column types
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    # Remove target if present
    if 'target' in numeric_cols:
        numeric_cols.remove('target')
    # heuristics for categoricals
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c != 'target']
    # numeric pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    # categorical pipeline (one-hot for low-cardinality)
    from sklearn.preprocessing import OneHotEncoder
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])
    return preprocessor, numeric_cols, categorical_cols

def prepare_and_split(path, target_col='target', test_size=0.2, random_state=42, out_processed='data/processed/train.npz'):
    df = load_data(path)
    # Basic cleaning example (customize to your dataset)
    df = df.drop_duplicates()
    # If dataset uses 'Class' like creditcard.csv:
    if target_col not in df.columns:
        if 'Class' in df.columns:
            df = df.rename(columns={'Class': target_col})
        elif 'default' in df.columns:
            df = df.rename(columns={'default': target_col})
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    preprocessor, num_cols, cat_cols = build_preprocessor(df, categorical_threshold=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=random_state)
    # Fit preprocessor on train
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    # Save preprocessor & column info
    joblib.dump({'preprocessor': preprocessor, 'num_cols': num_cols, 'cat_cols': cat_cols}, 'models/preprocessor.joblib')
    # Save processed arrays (numpy .npz)
    np.savez_compressed(out_processed, X_train=X_train_trans, X_test=X_test_trans, y_train=y_train.values, y_test=y_test.values)
    print("Saved processed data and preprocessor.")
    return X_train_trans, X_test_trans, y_train.values, y_test.values
