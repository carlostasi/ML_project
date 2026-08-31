import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path, keep_evaporation_proxy=False):
    """
    Load the raw dataset and apply feature engineering.

    No class balancing happens here: the whole dataset is returned with its
    natural class distribution, so that the train/test split downstream can
    carve out a test set that is representative of the real population.
    Balancing is applied to the training partition only, by
    balance_training_set.
    """
    df = pd.read_csv(file_path)

    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    df = features_engineering(df, keep_evaporation_proxy=keep_evaporation_proxy)
    target_col = "Irrigation_Need"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


def balance_training_set(X_train, y_train, sample_size_per_class, random_state=42):
    """
    Draw a class-balanced subsample of the TRAINING partition only.

    This must never be applied before the train/test split: balancing the whole
    dataset first would also balance the test set, and the resulting metrics
    would be measured on an artificial 1:1:1 distribution that does not exist in
    the field. Keeping the split first means every model is scored on the same
    test set, drawn from the real class distribution, whichever arena it was
    trained in.

    The per-class size is capped by the rarest class available in the training
    partition, so the requested sample_size_per_class acts as an upper bound.
    """
    counts = y_train.value_counts()
    n_per_class = min(sample_size_per_class, int(counts.min()))

    sampled_index = []
    for cls in sorted(y_train.unique()):
        cls_index = y_train.index[y_train == cls]
        sampled_index.append(
            pd.Series(cls_index).sample(n=n_per_class, random_state=random_state)
        )

    # Unify and shuffle, so that the class blocks are not left in order
    balanced_index = (
        pd.concat(sampled_index)
        .sample(frac=1, random_state=random_state)
        .to_numpy()
    )

    return X_train.loc[balanced_index], y_train.loc[balanced_index]

def features_engineering(df, keep_evaporation_proxy=False):
    """
    Trying to create new features based on climatic logic 
    to help the models to converge faster

    The evaporation proxy E is normally an intermediate term only: it measures
    thermal moisture extraction and reaches the design matrix through
    Water_Deficit, so atmospheric demand is represented once, in the ratio form
    that is agronomically meaningful. keep_evaporation_proxy=True retains it as
    a column of its own instead, which is the ablation the report promises in
    its feature-engineering section; see src/experiments/feature_ablation.py.
    """ 
    # Evaportaion risk index
    evap_proxy = df['Temperature_C'] / (df['Humidity'] + 1e-5)
    if keep_evaporation_proxy:
        df['Evaporation_proxy'] = evap_proxy
    # Daily thermo impact
    df['Thermal_impact'] = df['Temperature_C'] * df['Sunlight_Hours']

    # Soil Health (Carbonio Organico scalato sulla deviazione dal pH ottimale di 6.5)
    ph_deviation = np.abs(df['Soil_pH'] - 6.5)
    df['Soil_Health'] = np.log1p(df['Organic_Carbon'] / (ph_deviation + 1e-5))

    # Water Deficit (Rapporto tra domanda atmosferica e umidità reale del suolo)
    df['Water_Deficit'] = evap_proxy / (df['Soil_Moisture'] + 1e-5)

    return df


def get_pipeline_transformer(drop_numeric=None, add_numeric=None):
    """
    Define trasnformer pipeline for numerical and categorial columns.
    Apply One-Hot Ecncoding and Feature Scaling.

    drop_numeric and add_numeric exist for the feature ablation in
    src/experiments/feature_ablation.py and both default to no change, so every
    existing caller keeps the original 38-dimensional encoding. drop_numeric
    removes named continuous columns (used to take out the engineered terms one
    at a time); add_numeric appends columns that features_engineering only
    produces on request, currently just Evaporation_proxy.
    """
    numerical_cols = [
        'Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity', 
        'Temperature_C', 'Humidity', 'Sunlight_Hours', 
        'Wind_Speed_kmh', 'Field_Area_hectare',
        'Thermal_impact',
        'Soil_Health', 'Water_Deficit',
        'Rainfall_mm', 'Previous_Irrigation_mm'
        # , 'Evaporation_proxy'
    ]

    categorical_cols = [
        'Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season', 
        'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region'
    ]
    
    if drop_numeric:
        missing = set(drop_numeric) - set(numerical_cols)
        if missing:
            raise ValueError(f"drop_numeric names unknown columns: {sorted(missing)}")
        numerical_cols = [c for c in numerical_cols if c not in drop_numeric]

    if add_numeric:
        numerical_cols = numerical_cols + [c for c in add_numeric if c not in numerical_cols]

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