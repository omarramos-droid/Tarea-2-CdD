from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

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
    orig_distribution = y[target_col].value_counts()

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
    
def Fisher_model():
    from Script_DATA import preprocesar_datos
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, _ = preprocesar_datos(
        escalar_lda=True,
        balancear=True,
        sampling_strategy=0.5,
        random_state=13
    )
    
    fisher = LinearDiscriminantAnalysis(n_components=1)
    fisher.fit(X_train_scaled, y_train)
    
    y_pred = fisher.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    return {'modelo': fisher, 'matriz_confusion': cm, 'metricas': metrics}

def NaiveBayes():
    from Script_DATA import preprocesar_datos
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, _ = preprocesar_datos(
        escalar_lda=True,
        balancear=True,
        sampling_strategy=0.5,
        random_state=13
    )
    
    # Entrenamiento
    nb = GaussianNB()
    nb.fit(X_train_scaled, y_train)
    
    # Predicción y métricas
    y_pred = nb.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    return {'modelo': nb, 'matriz_confusion': cm, 'metricas': metrics}



def LDA_model():
    from Script_DATA import preprocesar_datos
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, _ = preprocesar_datos(
        escalar_lda=True,
        balancear=True,
        sampling_strategy=0.5,
        random_state=13
    )
    
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_scaled, y_train)
    
    y_pred = lda.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    return {'modelo': lda, 'matriz_confusion': cm, 'metricas': metrics}


def QDA_model():
    from Script_DATA import preprocesar_datos
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, _ = preprocesar_datos(
        escalar_lda=True,
        balancear=True,
        sampling_strategy=0.5,
        random_state=13
    )
    
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train_scaled, y_train)
    
    y_pred = qda.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    return {'modelo': qda, 'matriz_confusion': cm, 'metricas': metrics}

