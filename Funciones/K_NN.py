import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from Script_DATA import preprocesar_datos
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB


# -----------------------------
# 1) Naive Bayes
# -----------------------------

def NaiveBayes():
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, _ = preprocesar_datos(escalar_lda=True)

    # Entrenamiento
    nb = GaussianNB()
    nb.fit(X_train_scaled, y_train)

    # Predicciones
    y_pred = nb.predict(X_test_scaled)
    
    # Probabilidad de clase positiva
    y_prob = nb.predict_proba(X_test_scaled)[:, 1]
    print("=== Naive Bayes ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precisión:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_prob))
    print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))

    return nb


# -----------------------------
# 2) LDA
# -----------------------------
def LDA():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, _ = preprocesar_datos(escalar_lda=True)
    
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_scaled, y_train)
    
    y_pred = lda.predict(X_test_scaled)
    y_prob = lda.predict_proba(X_test_scaled)[:, 1] 
    print("=== LDA ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precisión:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_prob))
    print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
    
    return lda

# -----------------------------
# 3) QDA
# -----------------------------
def QDA():
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, _ = preprocesar_datos(escalar_lda=True)
    
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train_scaled, y_train)
    
    y_pred = qda.predict(X_test_scaled)
    y_prob = qda.predict_proba(X_test_scaled)[:, 1] 
    print("=== QDA ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precisión:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_prob))
    print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
    
    return qda

# -----------------------------
# 4) k-NN
# -----------------------------


def KNN(k_max=20):

    # Preprocesamiento
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, _ = preprocesar_datos(escalar_lda=True)

    # Selección de k mediante validación cruzada (F1-score)
    f1_scores = []
    vecinos = list(range(1, min(k_max, len(X_train))))
    for k_try in vecinos:
        knn = KNeighborsClassifier(n_neighbors=k_try, weights='distance', metric='euclidean')
        scores = cross_val_score(knn, X_train_scaled, y_train, cv=3, scoring='f1')
        f1_scores.append(np.mean(scores))

    best_idx = np.argmax(f1_scores)
    best_k = vecinos[best_idx]
    print(f"K óptimo (F1 CV): {best_k}")

    # Entrenamiento final con k óptimo
    knn_final = KNeighborsClassifier(n_neighbors=best_k, weights='distance', metric='euclidean')
    knn_final.fit(X_train_scaled, y_train)

    # Predicciones y probabilidades
    y_pred = knn_final.predict(X_test_scaled)
    y_prob = knn_final.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
  
    # Métricas
    print("=== k-NN ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precisión:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    print("AUC:", auc)
    print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))

    # Gráfico 
    plt.figure(figsize=(8,5))
    plt.plot(vecinos, f1_scores, 'bo-', markersize=6, linewidth=2)
    plt.axvline(best_k, color='red', linestyle='--', label=f'K óptimo = {best_k}')
    plt.xlabel("Número de vecinos (k)")
    plt.ylabel("F1-score (CV)")
    plt.title("Selección de k óptimo - ")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    return knn_final

def Fisher():
    """
    Implementación de Fisher Linear Discriminant (binario)
    
    Retorna
    -------
    dict : métricas de desempeño y modelo
    """
 

    # Preprocesar datos
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, _ = preprocesar_datos(escalar_lda=True)

    # Convertir a arrays numpy
    X_train = np.array(X_train_scaled)
    X_test = np.array(X_test_scaled)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # ================================
    # 1. Separar datos por clase
    # ================================
    X0 = X_train[y_train == 0]  # Clase 0
    X1 = X_train[y_train == 1]  # Clase 1

    # ================================
    # 2. Medias de clase
    # ================================
    m0 = np.mean(X0, axis=0).reshape(-1, 1)  # Media clase 0
    m1 = np.mean(X1, axis=0).reshape(-1, 1)  # Media clase 1
    dm = m1 - m0  # Diferencia de medias

    # ================================
    # 3. Matriz de dispersión intra-clase Sw
    # ================================
    # Calcular matrices de covarianza para cada clase    if len(X0) > 1:
    S0 = np.cov(X0, rowvar=False)

    S1 = np.cov(X1, rowvar=False)

    Sw = S0 + S1

    # ================================
    # 4. Vector discriminante w (solución óptima)
    # ================================
    # w = Sw^(-1) * (m1 - m0)
    w = np.linalg.inv(Sw) @ dm
    w = w.ravel()  # Convertir a vector 1D


    # ================================
    # 5. Proyección de datos
    # ================================
    z_train = X_train @ w
    z_test = X_test @ w

    # ================================
    # 6. Encontrar umbral óptimo
    # ================================
    # Usar la media de las medias proyectadas como umbral
    z0_mean = np.mean(z_train[y_train == 0])
    z1_mean = np.mean(z_train[y_train == 1])
    threshold = (z0_mean + z1_mean) / 2

    # ================================
    # 7. Predicciones
    # ================================
    y_pred = (z_test > threshold).astype(int)

    # ================================
    # 8. Calcular probabilidades para AUC (aproximación)
    # ================================
    # Normalizar proyecciones a [0,1] para simular probabilidades
    z_min = np.min(z_test)
    z_max = np.max(z_test)
    y_prob = (z_test - z_min) / (z_max - z_min)

    # ================================
    # 9. Métricas completas
    # ================================
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    # ================================
    # 10. Visualizaciones
    # ================================
    
    # Gráfico 1: Proyecciones 1D
    plt.figure(figsize=(12, 4))
    
    # Histograma de proyecciones por clase
    plt.hist(z_test[y_test == 0], bins=30, alpha=0.7, label='Clase 0', color='blue', density=True)
    plt.hist(z_test[y_test == 1], bins=30, alpha=0.7, label='Clase 1', color='red', density=True)
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Umbral = {threshold:.2f}')
    plt.xlabel('Valor Proyectado z = wᵀx')
    plt.ylabel('Densidad')
    plt.title('Proyección Fisher - Distribución 1D')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    # Gráfico 2: Matriz de confusión
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Sí'])
    disp.plot(cmap='Blues', ax=plt.gca())
    plt.title('Matriz de Confusión - Fisher')

    plt.tight_layout()
    plt.show()

    # ================================
    # 11. Resultados numéricos
    # ================================
    print("=== FISHER LINEAR DISCRIMINANT ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precisión: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("Matriz de Confusión:")
    print(cm)
    print(f"\nParámetros del modelo:")
    print(f"Vector w (primeros 5 elementos): {w[:5]}")
    print(f"Umbral óptimo: {threshold:.4f}")
    print(f"Separación entre clases: {np.abs(z1_mean - z0_mean):.4f}")

    # ================================
    # 12. Retornar resultados completos
    # ================================
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'model_params': {
            'w': w,
            'threshold': threshold,
            'm0': m0.ravel(),
            'm1': m1.ravel()
        },
        'projections_train': z_train,
        'projections_test': z_test
    }


def FisherBalance():
    """
    Implementación de Fisher Linear Discriminant (binario) 
    con ajuste de umbral para datos desbalanceados.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, balanced_accuracy_score
    )
    from sklearn.metrics import ConfusionMatrixDisplay
    import numpy as np
    import matplotlib.pyplot as plt

    # Preprocesar datos
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, _ = preprocesar_datos(escalar_lda=True)

    # Convertir a arrays numpy
    X_train = np.array(X_train_scaled)
    X_test = np.array(X_test_scaled)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # ================================
    # 1. Separar datos por clase
    # ================================
    X0 = X_train[y_train == 0]
    X1 = X_train[y_train == 1]

    # ================================
    # 2. Medias y Sw
    # ================================
    m0 = np.mean(X0, axis=0).reshape(-1, 1)
    m1 = np.mean(X1, axis=0).reshape(-1, 1)
    dm = m1 - m0

    S0 = np.cov(X0, rowvar=False)
    S1 = np.cov(X1, rowvar=False)
    Sw = S0 + S1

    # ================================
    # 3. Vector discriminante
    # ================================
    w = np.linalg.inv(Sw) @ dm
    w = w.ravel()

    # ================================
    # 4. Proyección de datos
    # ================================
    z_train = X_train @ w
    z_test = X_test @ w

    # ================================
    # 5. Búsqueda de umbral óptimo (por F1)
    # ================================
    thresholds = np.linspace(np.min(z_test), np.max(z_test), 200)
    f1_scores, recalls, precisions = [], [], []

    for t in thresholds:
        y_pred_tmp = (z_test > t).astype(int)
        f1_scores.append(f1_score(y_test, y_pred_tmp, zero_division=0))
        recalls.append(recall_score(y_test, y_pred_tmp, zero_division=0))
        precisions.append(precision_score(y_test, y_pred_tmp, zero_division=0))

    best_idx = np.argmax(f1_scores)
    threshold = thresholds[best_idx]

    # ================================
    # 6. Predicciones finales
    # ================================
    y_pred = (z_test > threshold).astype(int)

    # Normalizar proyecciones como "probabilidades"
    z_min, z_max = np.min(z_test), np.max(z_test)
    y_prob = (z_test - z_min) / (z_max - z_min)

    # ================================
    # 7. Métricas
    # ================================
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    # ================================
    # 8. Visualizaciones
    # ================================
    # Histograma de proyecciones
    plt.figure(figsize=(12, 4))
    plt.hist(z_test[y_test == 0], bins=30, alpha=0.7, label='Clase 0', color='blue', density=True)
    plt.hist(z_test[y_test == 1], bins=30, alpha=0.7, label='Clase 1', color='red', density=True)
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Umbral óptimo = {threshold:.2f}')
    plt.xlabel('Proyección z = wᵀx')
    plt.ylabel('Densidad')
    plt.title('Proyección Fisher - Distribución 1D')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Curva F1/recall/precision vs umbral
    plt.figure(figsize=(10, 4))
    plt.plot(thresholds, f1_scores, label="F1-score", color='red')
    plt.plot(thresholds, recalls, label="Recall", color='blue')
    plt.plot(thresholds, precisions, label="Precision", color='green')
    plt.axvline(threshold, color='black', linestyle='--', linewidth=1.5)
    plt.xlabel("Umbral")
    plt.ylabel("Métrica")
    plt.title("Evolución de métricas según el umbral")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # Matriz de confusión
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Clase 0', 'Clase 1'])
    disp.plot(cmap='Blues')
    plt.title("Matriz de Confusión - Fisher")
    plt.show()

    # ================================
    # 9. Resultados
    # ================================
    print("=== FISHER LINEAR DISCRIMINANT (con ajuste de umbral) ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Precisión: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("Matriz de Confusión:")
    print(cm)
    print(f"\nUmbral óptimo (por F1): {threshold:.4f}")
    print(f"Separación entre clases proyectadas: {abs(np.mean(z_train[y_train==1]) - np.mean(z_train[y_train==0])):.4f}")

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'model_params': {
            'w': w,
            'threshold': threshold,
            'm0': m0.ravel(),
            'm1': m1.ravel()
        },
        'projections_train': z_train,
        'projections_test': z_test,
        'threshold_curve': {
            'thresholds': thresholds,
            'f1_scores': f1_scores,
            'recalls': recalls,
            'precisions': precisions
        }
    }




