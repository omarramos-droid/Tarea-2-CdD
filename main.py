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



if __name__ == "__main__":
    resultados = KNN()
    print(resultados)
    print("=== RESULTADOS KNN ===")
    print(f"Mejor k: {resultados['k_optimo']}")
    print(f"F1 en validación (CV): {resultados['f1_validacion']:.4f}")
    print(f"F1 en test: {resultados['f1_test']:.4f}")
    print(f"Accuracy: {resultados['accuracy_test']:.4f}")
    print(f"Precision en test: {resultados['precision_test']:.4f}") 
    print("Matriz de confusión:")
    print(resultados['matriz_confusion'])
