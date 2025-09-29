import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def KNN():
    """
    K-NN optimizado con búsqueda en malla y SMOTE
    - Devuelve métricas clave en un diccionario
    """
    from Script_DATA import preprocesar_datos
    
    # Carga de datos con submuestreo y escalado
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, _ = preprocesar_datos(
        escalar_lda=True, 
        balancear=True,
        sampling_strategy=0.5,
        random_state=13
    )
    
    # Pipeline KNN + SMOTE
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=13)),
        ('knn', KNeighborsClassifier(n_jobs=-1))
    ])
    
    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 20, 25, 30],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['manhattan', 'euclidean']
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='f1',  # usamos F1 como métrica de optimización
        n_jobs=-1,
        return_train_score=True
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    # Mejor modelo
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    
    # Métricas
    f1_test = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Gráfico 1: evolución de F1 con k
    resultados = grid_search.cv_results_
    k_scores = {}
    for i, params in enumerate(resultados['params']):
        k = params['knn__n_neighbors']
        score = resultados['mean_test_score'][i]
        if k not in k_scores:
            k_scores[k] = []
        k_scores[k].append(score)

    k_valores = sorted(k_scores.keys())
    k_promedios = [np.mean(k_scores[k]) for k in k_valores]
    best_k_idx = np.argmax(k_promedios)
    best_k = k_valores[best_k_idx]
    best_k_score = k_promedios[best_k_idx]

    plt.figure(figsize=(10,6))
    plt.plot(k_valores, k_promedios, 'bo-', linewidth=2, markersize=8, label='F1-Score promedio')
    plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'k óptimo = {best_k}')
    plt.axhline(y=best_k_score, color='green', linestyle='--', alpha=0.5, label=f'F1 = {best_k_score:.4f}')
    plt.xlabel('Número de vecinos (k)')
    plt.ylabel('F1-Score promedio (validación cruzada)')
    plt.title('Selección óptima del parámetro k en K-NN', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Gráfico 2: matriz de confusión
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['no', 'yes'])
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d', colorbar=False)
    plt.title('Matriz de Confusión - KNN')
    plt.tight_layout()
    plt.show()
    
    return {
        'k_optimo': best_k, 
        'f1_validacion': best_k_score,
        'f1_test': f1_test,
        'accuracy_test': accuracy,
        'precision_test': precision,
        'recall_test': recall,
        'modelo': best_model,
        'matriz_confusion': cm
    }
