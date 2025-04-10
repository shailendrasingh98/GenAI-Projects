import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic preprocessing on a pandas DataFrame:
    1. Identifies numerical and categorical columns.
    2. Imputes missing values (median for numerical, mode for categorical).
    3. Scales numerical features using StandardScaler.
    4. One-hot encodes categorical features.

    Args:
        df: The input pandas DataFrame.

    Returns:
        A new pandas DataFrame with preprocessing applied.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    # Separate features and target if applicable (assuming no specific target column for now)
    # If you have a target column, you might want to separate it before preprocessing
    # X = df.drop('target_column', axis=1)
    # y = df['target_column']
    X = df.copy()

    # Identify column types
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Create preprocessing pipelines for numerical and categorical features
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Create a column transformer to apply pipelines to the correct columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ],
        remainder='passthrough' # Keep other columns (if any) untouched
    )

    # Apply the preprocessing steps
    X_processed = preprocessor.fit_transform(X)

    # Get feature names after one-hot encoding
    # Handle cases where there are no categorical or numerical columns
    feature_names = []
    if numerical_cols:
        feature_names.extend(numerical_cols)

    if categorical_cols:
      try:
          # Access the fitted OneHotEncoder to get feature names
          ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
          feature_names.extend(ohe_feature_names)
      except AttributeError:
           # Handle older scikit-learn versions if necessary or if no categorical columns existed
           pass


    # Add names for columns passed through via 'remainder'
    try:
        remainder_cols = X.columns.difference(numerical_cols + categorical_cols)
        if not remainder_cols.empty:
             feature_names.extend(remainder_cols)
    except Exception:
        pass # Handle potential errors if column sets are empty


    # Reconstruct the DataFrame
    # Check if X_processed is empty before creating DataFrame
    if X_processed.shape[1] == 0:
        processed_df = pd.DataFrame()
    elif len(feature_names) == X_processed.shape[1]:
         processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
    else:
         # Fallback if feature name generation failed or mismatched
         processed_df = pd.DataFrame(X_processed, index=X.index)
         print("Warning: Could not reliably determine output feature names. Using default numerical column names.")


    return processed_df
