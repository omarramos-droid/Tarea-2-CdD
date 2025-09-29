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





