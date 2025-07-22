import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("CSV file is empty.")
        print("✅ Data loaded successfully.")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def preprocess_data(df):
    target_column = 'target' if 'target' in df.columns else None
    X = df.drop(columns=[target_column]) if target_column else df
    y = df[target_column] if target_column else None


    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()


    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

   
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

  
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    X_transformed = preprocessor.fit_transform(X)
    print("✅ Data transformation complete.")
    return X_transformed, y

def save_clean_data(X, y=None, output_dir='clean_data'):
    os.makedirs(output_dir, exist_ok=True)

 
    X_df = pd.DataFrame(X)
    X_df.to_csv(os.path.join(output_dir, 'features.csv'), index=False)

 
    if y is not None:
        y_df = pd.DataFrame(y, columns=['target']) if isinstance(y, pd.Series) else pd.DataFrame(y)
        y_df.to_csv(os.path.join(output_dir, 'target.csv'), index=False)

    print(f"✅ Clean data saved to '{output_dir}/'")

def run_pipeline(input_file):
    df = load_data(input_file)
    if df is not None:
        X, y = preprocess_data(df)
        save_clean_data(X, y)

if __name__ == "__main__":
    input_file_path = 'data.csv'
    run_pipeline(input_file_path)
