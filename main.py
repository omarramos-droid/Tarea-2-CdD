

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import scipy.linalg
import time

#Agregar la ruta de funciones auxiliares al path de Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'Funciones'))
from script_rodolfo import *
from Script_DATA import *
from K_NN import *





if __name__ == "__main__":
    print('------------------ Parte I ------------------')
    print('Tarda de 4 a 5 min en correr')
    #Crea datos y carpeta
    gdatos()  
# Estas funciones tardan en promedio 3 minutos en implementarse
# Arogando 8 imagenes, las cuales están en el reporte
    nb_model = NaiveBayes()
    lda_model = LDA()
    qda_model = QDA()
    #knn_model = KNN()
    fisher_model=Fisher()
    fisher_model_balance=FisherBalance()
    

     
    # ===== Casos que se pueden correr =====
    print('------------------ Parte II ------------------')
    print('Tarda de 1 a 2 min en correr')
    '''
    Casos:
        misma-dificil
        diferente-dificil
        misma-alta_correlacion
        diferente_alta_correlacion
    '''
    
    caso = 'misma-alta_correlacion'
    
    # Diccionario de casos de hiperparámetros
    casos_normales = casos_norm()
    
    # Hacemos la simulación de el errore real de bayes y los errores de los demás clasificadores
    df_final, df_final_mean = simulacion_todos_los_metodos(caso, casos_normales )
    
    
    # gráfico de L(g) vs todos los modelos y variando el desbalance
    grafica_lg_vs_clasificadores(df_final_mean)
    
    # grafica de L(g) vs los modelos KNN y variando el desbalance
    grafica_lg_vs_knn_desbalance_fijo(df_final_mean)
    
    # grafica de L(g) vs los modelos KNNs
    grafica_ls_vs_knns(df_final_mean)
    
    # grafica heatmap de diferencias 
    heatmapdiff(df_final_mean)
    
    # grafico de puntos y aprendizaje
    #grafica_puntos_y_aprendizaje_qda(50, 150,caso ,  casos_norm)
    
    # grafico de puntos y aprendizaje
    #grafica_puntos_y_aprendizaje_nb(50, 50,caso ,  casos_normales)
    
    # grafico de puntos y aprendizaje
    #grafica_puntos_y_aprendizaje_knn(50, 50,caso ,  casos_normales)
    
    
    

  