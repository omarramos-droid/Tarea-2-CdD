import os
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def gdatos():
    os.makedirs("data", exist_ok=True)

# Descargar dataset Bank Marketing (ID = 222 en UCI)
    bank_marketing = fetch_ucirepo(id=222)
  
# fetch dataset 
    bank_marketing = fetch_ucirepo(id=222) 
  
# data (as pandas dataframes) 
    X = bank_marketing.data.features 
    y = bank_marketing.data.targets 

# Separar características y etiquetas
    X = bank_marketing.data.features
    y = bank_marketing.data.targets

# Save data
    X.to_csv("data/features.csv", index=False)
    y.to_csv("data/targets.csv", index=False)
def preprocesar_datos(
    escalar_lda=False, test_size=0.3, random_state=13, balancear=False, sampling_strategy=0.5):
    """
    Preprocesa datos para clasificadores, con opción de balancear clases.

    Parameters
    ----------
    escalar_lda : bool, optional
        Si es True, retorna también datos escalados para LDA/QDA
    test_size : float, optional
        Proporción de datos para test
    random_state : int, optional
        Semilla para reproducibilidad
    balancear : bool, optional
        Si es True, aplica submuestreo moderado para balancear clases
    sampling_strategy : float, optional
        Proporción de la clase mayoritaria a mantener (0.5 = 50%)
    
    Returns
    -------
    tuple
        Dependiendo de escalar_lda, retorna diferentes conjuntos de datos
    """
    X = pd.read_csv("data/features.csv")
    y = pd.read_csv("data/targets.csv")

    # Reset indices para asegurar alineación
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Eliminar variables con problemas
    variables_excluir = ['contact', 'duration']  # contact: >13k NA, duration: leakage
    X = X.drop(columns=variables_excluir, errors='ignore')

    # --- IMPUTACIÓN SEGÚN EL MECANISMO DE FALTANTES ---
    # job y education -> MAR (rellenar con 'unknown')
    for var in ['job', 'education']:
        if var in X.columns:
            X[var] = X[var].fillna("unknown")

    # poutcome -> MNAR (faltantes significan "no contactado")
    if "poutcome" in X.columns:
        X["poutcome"] = X["poutcome"].fillna("no_contacted")

    # --- TRANSFORMACIÓN DE LA VARIABLE OBJETIVO ---
    target_col = y.columns[0]
    y_binary = (y[target_col] == 'yes').astype(int)

    # Variables categóricas
    vars_one_hot = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome']
    vars_label = ['month', 'day_of_week']

    # Copia para procesar
    X_encoded = X.copy()

    # Label Encoding (para variables ordinales)
    label_encoders = {}
    for var in vars_label:
        if var in X_encoded.columns:
            le = LabelEncoder()
            X_encoded[var] = le.fit_transform(X_encoded[var])
            label_encoders[var] = le

    # One-Hot Encoding (para categóricas nominales)
    X_encoded = pd.get_dummies(
        X_encoded, 
        columns=vars_one_hot, 
        prefix=vars_one_hot, 
        drop_first=True
    )

    # --- BALANCEO DE CLASES ---
    if balancear:
        from imblearn.under_sampling import RandomUnderSampler
        under_sampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )
        X_encoded, y_binary = under_sampler.fit_resample(X_encoded, y_binary)

    # --- TRAIN/TEST SPLIT ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_binary,
        test_size=test_size,
        random_state=random_state,
        stratify=y_binary
    )

    # --- ESCALADO PARA LDA/QDA ---
    if escalar_lda:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled = pd.DataFrame(
            X_train_scaled, columns=X_train.columns, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled, columns=X_test.columns, index=X_test.index
        )

        return (X_train, X_test, y_train, y_test,
                X_train_scaled, X_test_scaled, label_encoders)

    return X_train, X_test, y_train, y_test, label_encoders









