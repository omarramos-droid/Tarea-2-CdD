import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import scipy.linalg
import time
from script_rodolfo import install_package

#Agregar la ruta de funciones auxiliares al path de Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'Funciones'))
from Script_DATA import *
from K_NN import *



if __name__ == "__main__":
    
# Estas funciones tardan en promedio 3 minutos en implementarse
# Arogando 8 imagenes, las cuales están en el reporte
    nb_model = NaiveBayes()
    lda_model = LDA()
    qda_model = QDA()
    knn_model = KNN()
    fisher_model=Fisher()
    fisher_model_balance=FisherBalance()
    
    
    
    

  