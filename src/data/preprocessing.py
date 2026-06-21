from sklearn.compose import _column_transformer
import collections
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path, sample_size_per_class=35000):
    df = pd.read_csv(file_path)

    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    df = features_engineering(df)
    target_col = "Irrigation_Need"

    if sample_size_per_class is None:
        # Big Data here (whole dataset)
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return X, y

    # Trying to balance here
    classes = df[target_col].unique()
    sampled_dfs = []

    for cls in classes:
        df_cls = df[df[target_col] == cls]
        n_samples = min(sample_size_per_class, len(df_cls))
        df_sampled_cls = df_cls.sample(n=n_samples, random_state=42)
        sampled_dfs.append(df_sampled_cls)

    # Here we unify and randomize the balanced dataset
    df_balanced = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

    X = df_balanced.drop(columns=[target_col])
    y = df_balanced[target_col]

    return X, y

def features_engineering(df):
    """
    Trying to create new features based on climatic logic 
    to help the models to converge faster
    """ 
    # Total water stored in the terrain
    df['Total_water_input'] = df['Rainfall_mm'] + df['Previous_Irrigation_mm']
    # Evaportaion risk index
    df['Evaporation_proxy'] = df['Temperature_C'] / (df['Humidity'] + 1e-5)
    # Daily thermo impact
    df['Thermal_impact'] = df['Temperature_C'] * df['Sunlight_Hours']

    return df


def get_pipeline_transformer():
    """
    Define trasnformer pipeline for numerical and categorial columns.
    Apply One-Hot Ecncoding and Feature Scaling.
    """
    numerical_cols = [
        'Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity', 
        'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours', 
        'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm',
        'Total_water_input', 'Evaporation_proxy', 'Thermal_impact'
    ]

    categorical_cols = [
        'Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season', 
        'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region'
    ]
    
    # Construction of ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ]
    )

    return preprocessor

def dataset_setup(X, y, test_size=0.2):
    # Map multi-label target
    target_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    y_numeric = y.map(target_mapping)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_numeric, test_size=test_size, stratify=y_numeric, random_state=42
    )

    return X_train, X_test, y_train, y_test