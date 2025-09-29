import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.linalg
import time
from script_rodolfo import install_package

# Agregar la ruta de funciones auxiliares al path de Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'Funciones'))
from Script_DATA import *
from K_NN import *
# from Clasificadores import *



if __name__ == "__main__":
    
    
    nb_model = NaiveBayes()
    lda_model = LDA()
    qda_model = QDA()
    knn_model = KNN()
    
    
    
    
    
    
    
    # resultados = KNN()
    # print("=== RESULTADOS KNN ===")
    # print(f"Mejor k: {resultados['k_optimo']}")
    # print(f"F1 en test: {resultados['f1_test']:.4f}")

    # print(resultados['matriz_confusion'])
    # Resultados_NB=NaiveBayes()
    # print("=== RESULTADOS NB ===")
    # print(f"Accuracy: {Resultados_NB['metricas']['accuracy'] } " )
    # print(f"Precision: {Resultados_NB['metricas']['precision'] } " )
    # print(f"f1: {Resultados_NB['metricas']['f1'] } " )
    # # Resultados_LDA=LDA_model()
    # print("=== RESULTADOS LDA ===")
    # print(f"Accuracy: {Resultados_LDA['metricas']['accuracy'] } " )
    # print(f"Precision: {Resultados_LDA['metricas']['precision'] } " )
    # print(f"f1: {Resultados_LDA['metricas']['f1'] } " )
    # Resultados_QDA=QDA_model()
    # print("=== RESULTADOS QDA ===")
    # print(f"Accuracy: {Resultados_QDA['metricas']['accuracy'] } " )
    # print(f"Precision: {Resultados_QDA['metricas']['precision'] } " )
    # print(f"f1: {Resultados_QDA['metricas']['f1'] } " )  
    
    
  