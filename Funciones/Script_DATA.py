import os
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def descargar_datos():
    """
    Descarga y guarda el dataset Bank Marketing del repositorio UCI
    """
    os.makedirs("data", exist_ok=True)
        
    bank_marketing = fetch_ucirepo(id=222)
        
    X = bank_marketing.data.features 
    y = bank_marketing.data.targets 

    X.to_csv("data/features.csv", index=False)
    y.to_csv("data/targets.csv", index=False)
    
    # Mostrar distribución original del target
    print("Distribución original del target:")
    print(y.iloc[:, 0].value_counts())
    print(f"Total de registros: {len(y)}")
    
    
def preprocesar_datos(escalar_lda=False, test_size=0.3, random_state=13, balancear=False, sampling_strategy=0.5):
    """
    Preprocesa datos para clasificadores, con opción de balancear clases
    
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
    
    # Mostrar distribución antes de procesar
    target_col = y.columns[0]
    print("\n=== DISTRIBUCIÓN ORIGINAL ===")
    orig_distribution = y[target_col].value_counts()
    print(orig_distribution)
    print(f"Total registros: {len(y)}")
    print(f"Proporción Yes: {orig_distribution.get('yes', 0)/len(y)*100:.1f}%")
    
    variables_excluir = ['contact', 'duration']
    X = X.drop(columns=variables_excluir, errors='ignore')
   
    # Convertir target a binario (0/1)
    y_binary = (y[target_col] == 'yes').astype(int)
    
    # Variables para codificación
    vars_one_hot = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome']
    vars_label = ['month', 'day_of_week']
    
    # Crear copia para procesamiento
    X_encoded = X.copy()
    
    # Aplicar Label Encoding a variables ordinales
    label_encoders = {}
    for var in vars_label:
        if var in X_encoded.columns:
            le = LabelEncoder()
            X_encoded[var] = le.fit_transform(X_encoded[var])
            label_encoders[var] = le
    
    # Aplicar One-Hot Encoding a variables nominales
    X_encoded = pd.get_dummies(X_encoded, columns=vars_one_hot, prefix=vars_one_hot, drop_first=True)
    
   
    # BALANCEAR CON SUBMUESTREO MODERADO SI SE SOLICITA
    if balancear:
        from imblearn.under_sampling import RandomUnderSampler
        
        # Aplicar submuestreo ANTES de la división train/test
        under_sampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy, 
            random_state=random_state
        )
        X_balanced, y_balanced = under_sampler.fit_resample(X_encoded, y_binary)
       
        # Usar los datos balanceados
        X_encoded = X_balanced
        y_binary = y_balanced
    
    # División estratificada train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_binary, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y_binary
    )
 
    if escalar_lda:
        # Escalar datos para LDA/QDA
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convertir a DataFrame para mantener nombres de columnas
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        return (X_train, X_test, y_train, y_test, 
                X_train_scaled, X_test_scaled, label_encoders)
    
    return X_train, X_test, y_train, y_test, label_encoders
    
def obtener_estadisticas_dataset(X_train, X_test, y_train, y_test):
    """
    Genera estadísticas del dataset procesado
    
    Parameters
    ----------
    X_train, X_test : DataFrame
        Conjuntos de entrenamiento y prueba
    y_train, y_test : Series
        Variables objetivo
        
    Returns
    -------
    dict
        Diccionario con estadísticas del dataset
    """
    stats = {
        'n_entrenamiento': X_train.shape[0],
        'n_prueba': X_test.shape[0],
        'n_variables': X_train.shape[1],
        'prop_clase_1_entrenamiento': np.mean(y_train),
        'prop_clase_1_prueba': np.mean(y_test),
        'balance_entrenamiento': f"{sum(y_train == 0)}:{sum(y_train == 1)}",
        'balance_prueba': f"{sum(y_test == 0)}:{sum(y_test == 1)}"
    }
   
    

    return stats




