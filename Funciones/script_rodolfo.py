import subprocess
import sys
import importlib




import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from functools import reduce


def install_package(package_name):
    """Instala un paquete si no está disponible"""
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name} ya está instalado")
    except ImportError:
        print(f"📦 Instalando {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✅ {package_name} instalado correctamente")
        except subprocess.CalledProcessError:
            print(f"❌ Error instalando {package_name}")
            sys.exit(1)
            
def grafica_puntos(df, nombre):
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x="x1", y="x2", hue="cat",
                    palette=['red', 'blue'], s=70, edgecolor="black")
    plt.title(nombre, fontsize=14)
    plt.xlabel("Variable X1")
    plt.ylabel("Variable X2")
    plt.legend(title="Clase")
    plt.grid(alpha=0.3)
    plt.show()
    



def clasificador_optimo_bayes(x, dictio_mean_sigma_pi):
    
    evaluacion = [dictio_mean_sigma_pi[i]['densidad'](x) * dictio_mean_sigma_pi[i]['pi']\
         for i in list(dictio_mean_sigma_pi.keys())]
    
    return list(dictio_mean_sigma_pi.keys())[np.argmax(evaluacion)]

def metri_L(X, dictio_mean_sigma_pi):
    cova = np.array(X[['x1', 'x2']])
    ypred = [clasificador_optimo_bayes(i, dictio_mean_sigma_pi ) for i in cova]
    ypred = np.array([0 if i =='cat0' else 1 for i in ypred])
    return accuracy_score(X['cat'], ypred)


def plot_regiones_COB(X , y, dictio_mean_sigma_pi, title):
    
    # Crear nueva figura cada vez
    plt.figure(figsize=(6, 5))
    
    # Crear grid en el espacio 2D
    x_min, x_max = X['x1'].min() - 1, X['x1'].max() + 1
    y_min, y_max = X['x2'].min() - 1, X['x2'].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Predicciones sobre el grid
    Z = [clasificador_optimo_bayes(i, dictio_mean_sigma_pi ) for i in np.c_[xx.ravel(), yy.ravel()]]
    Z = [0 if i =='cat0' else 1 for i in Z]
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)

    # Graficar fronteras y puntos
    cmap = ListedColormap(['red', 'blue'])
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    plt.scatter(X['x1'], X['x2'], c=y, cmap=cmap, edgecolor="k", s=40)
    plt.title(title)
    plt.show()
    
    
def simulacion(dict_casos, caso, n0, n1):
    vector_datos = dict_casos[caso]
    mu0 = vector_datos['mu0']
    mu1 = vector_datos['mu1']
    sigma0 = vector_datos['sigma0']
    sigma1 = vector_datos['sigma1']
    
    simn0 = np.random.multivariate_normal(mean=mu0, cov=sigma0, size = n0)
    simn1 = np.random.multivariate_normal(mean=mu1, cov=sigma1, size = n1)
    
    dfsimn0 = pd.DataFrame(simn0, columns=['x1', 'x2'])
    dfsimn0['cat'] = 0
    
    dfsimn1 = pd.DataFrame(simn1, columns=['x1', 'x2'])
    dfsimn1['cat'] = 1
    df = pd.concat([dfsimn0, dfsimn1], axis=0).reset_index(drop=True)
    return df, mu0, mu1, sigma0, sigma1
    

def casos_norm():
    p = 2
    casos =  {
        'misma-facil': {
            'mu0' : np.array([-10, 10]),
            'mu1': np.array([10, -10]),
            'sigma0': [[1 if i == j else 0 for j in range(p)] for i in range(p)],
            'sigma1': [[1 if i == j else 0 for j in range(p)] for i in range(p)]
        },
        'misma-intermedio': {
            'mu0' : np.array([-1, 1]),
            'mu1': np.array([1, -1]),
            'sigma0': [[1 if i == j else 0 for j in range(p)] for i in range(p)],
            'sigma1': [[1 if i == j else 0 for j in range(p)] for i in range(p)]
        },
        'misma-dificil': {
            'mu0' : np.array([-1, 1]),
            'mu1': np.array([1, -1]),
            'sigma0': [[2 if i == j else 0 for j in range(p)] for i in range(p)],
            'sigma1': [[2 if i == j else 0 for j in range(p)] for i in range(p)]
        },
        'diferente-facil': {
            'mu0' : np.array([-10, 10]),
            'mu1': np.array([10, -10]),
            'sigma0': [[1 if i == j else 0 for j in range(p)] for i in range(p)],
            'sigma1': [[4 if i == j else 0 for j in range(p)] for i in range(p)]
        },
        'diferente-intermedio': {
            'mu0' : np.array([-2, 2]),
            'mu1': np.array([2, -2]),
            'sigma0': [[1 if i == j else 0 for j in range(p)] for i in range(p)],
            'sigma1': [[3 if i == j else 0 for j in range(p)] for i in range(p)]
        },
        'diferente-dificil': {
            'mu0' : np.array([-2, 2]),
            'mu1': np.array([0, 0]),
            'sigma0': [[1 if i == j else 0 for j in range(p)] for i in range(p)],
            'sigma1': [[4 if i == j else 0 for j in range(p)] for i in range(p)]
        },
        'misma-alta_correlacion':{
            'mu0' : np.array([-2, 2]),
            'mu1': np.array([0, 0]),
            'sigma0':[[4.0 , 5.7], [5.7 ,9.0 ]],
            'sigma1': [[4.0 , 5.7], [5.7 ,9.0 ]]
        },
        'diferente_alta_correlacion':{
            'mu0' : np.array([-2, 2]),
            'mu1': np.array([0, 0]),
            'sigma0':[[4.0 , 5.7], [5.7 ,9.0 ]],
            'sigma1': [[ 1.0, -3.6], [-3.6, 16.0 ]]
            }
        
    }
    return casos


def sim_error_clasif_opt_bayes(tipo_clases, casos_norm ):
    tipo_clases = [tipo_clases]
    info_optbayes_sim = dict()
    for caso in tipo_clases:
        info_optbayes_sim[caso] = dict()
        for n0 in [50]:
            info_optbayes_sim[caso][f'n0_{n0}'] = dict()
            for n1 in [50, 100, 150, 200,250, 300, 350]:
                info_optbayes_sim[caso][f'n0_{n0}'][f'n1_{n1}'] = dict()
                pi0 = n0/(n0+n1)
                pi1 = 1- pi0
                lsbayes = []
                
                for n in range(100):
                    #Genera el data Frame con las simulaciones
                    df, mu0, mu1, sigma0, sigma1 = simulacion(casos_norm, caso, n0, n1) 
                    X = df[['x1', 'x2']]
                    y = df['cat']
                    
                    # --- Clasificador Óptimo de Bayes --- 
                    
                    # Lista con los datos para calcular el error del clasificador de bayes
                    dict_densities = {'cat0': {'densidad':lambda x: multivariate_normal.pdf(x, mean=mu0,
                                                                    cov=sigma0), 'pi':pi0} ,
                                      'cat1': {'densidad':lambda x: multivariate_normal.pdf(x, mean=mu1,
                                                                   cov=sigma1), 'pi':pi1}}
                    #Error de Bayes
                    Lbayes = 1-metri_L(df, dict_densities)
                    
                    
                    
                    #Listas de Error
                    lsbayes += [Lbayes]
                    
                    
                info_optbayes_sim[caso][f'n0_{n0}'][f'n1_{n1}']['bayes'] = {'sim' : lsbayes,
                                                        'mean' : np.mean(lsbayes),
                                                        'std_inf' : np.mean(lsbayes)-np.std(lsbayes), 
                                                        'std_sup' : np.mean(lsbayes)+np.std(lsbayes)}
                
                print(f'fin-{caso}-_n0_{n0}_y_n1_{n1}')
                
    return info_optbayes_sim
    
    

def simulacion_todos_los_metodos(tipo_clases, casos_norm ):
    
    info_optbayes_sim = sim_error_clasif_opt_bayes(tipo_clases, casos_norm )
    
    caso = tipo_clases
    n0 =50
    modelo = 'bayes'
    ls = []
    for n1 in [50, 100, 150, 200,250, 300, 350]:            
        mean = info_optbayes_sim[caso][f'n0_{n0}'][f'n1_{n1}'][modelo]['mean']
        sup = info_optbayes_sim[caso][f'n0_{n0}'][f'n1_{n1}'][modelo]['std_sup']
        dvs =  sup - mean
        ls += [[modelo ,n1, mean, dvs]]
        
    df_optbayes = pd.DataFrame(ls, columns = ['modelo','num_datos', 'mean', 'dvs'])
    df_optbayes = df_optbayes.pivot(index='num_datos', columns='modelo', values=['mean', 'dvs'])
    
    #Cross Validation
    info_sim = dict()
    for caso in [tipo_clases]:
        info_sim[caso] = dict()
        for n0 in [50]:
            info_sim[caso][f'n0_{n0}'] = dict()
            for n1 in [50, 100, 150, 200,250, 300, 350]:
                info_sim[caso][f'n0_{n0}'][f'n1_{n1}'] = dict()
                pi0 = n0/(n0+n1)
                pi1 = 1- pi0
            
                
                df, mu0, mu1, sigma0, sigma1 = simulacion(casos_norm, caso, n0, n1) 
                X = df[['x1', 'x2']]
                y = df['cat']
                
                
                models = {
                "Naive Bayes": GaussianNB(priors=[pi0, pi1]),
                "LDA": LinearDiscriminantAnalysis(priors=[pi0, pi1]),
                "QDA": QuadraticDiscriminantAnalysis(priors=[pi0, pi1]),
                "k-NN (k=1)": KNeighborsClassifier(n_neighbors=1),
                "k-NN (k=3)": KNeighborsClassifier(n_neighbors=3),
                "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
                "k-NN (k=11)": KNeighborsClassifier(n_neighbors=11),
                "k-NN (k=21)": KNeighborsClassifier(n_neighbors=21),
                }
                
                cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2, )
    
                for name, model in models.items():
                    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
                    info_sim[caso][f'n0_{n0}'][f'n1_{n1}'][name] = {'scores':scores, 
                                                                    'mean': (1-scores).mean(),
                                                                    'dvs': np.std(1-scores)}
        
    ls_dfs = []
    for modelo in ['Naive Bayes', 'LDA',
                   'QDA', 'k-NN (k=1)', 
                   'k-NN (k=3)','k-NN (k=5)', 
                   'k-NN (k=11)', 'k-NN (k=21)']:            
        caso = tipo_clases
        n0 =50
        ls = []
        for n1 in [50, 100, 150, 200,250, 300, 350]:            
            mean = info_sim[caso][f'n0_{n0}'][f'n1_{n1}'][modelo]['mean']
            dvs = info_sim[caso][f'n0_{n0}'][f'n1_{n1}'][modelo]['dvs']
            ls += [[modelo ,n1, mean, dvs]]
            
        df = pd.DataFrame(ls, columns = ['modelo','num_datos',  'mean', 'dvs'])
        df = df.pivot(index='num_datos', columns='modelo', values=['mean', 'dvs'])
        ls_dfs += [df]

    df_otros = reduce(lambda x, y: x.join(y), ls_dfs)
    df_final = df_otros.join(df_optbayes)
    df_final.columns = ['{}-{}'.format(a, b) for a, b in df_final.columns]
    mean_cols = [i for i in df_final.columns if i[:4] == 'mean']
    df_final_mean = df_final[mean_cols]
    
    return df_final, df_final_mean

def grafica_lg_vs_clasificadores(df):
    plt.figure(figsize=(10,6))
    for col in df.columns:
        if col == "mean-bayes":
            plt.plot(df.index, df[col], linestyle=":", linewidth=2.5,
                     marker="o", markersize=6, label='Clasif opt. de Bayes', color="black")
        else:
            plt.plot(df.index, df[col], marker="o",
                     markersize=6, label=col[5:])
    
    # Ajustes de estilo
    plt.xlabel("Tamaño Población 1")
    plt.ylabel("L(g)")
    plt.title("Comparación de clasificadores con #Pob0 =50")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
    
def grafica_lg_vs_knn_desbalance_fijo(df, pob2 = 150):
    df_k = pd.DataFrame(df[[i for i in df.columns if i[:6] == 'mean-k']].loc[pob2])
    df_k.columns = ['L(g)']
    df_k['k'] = [1,3,5,11,21]
    df_k = df_k.reset_index(drop=True)
    
    plt.figure(figsize=(8,5))
    plt.plot(df_k['k'].astype(str), df_k['L(g)'], marker='o', linestyle='-', color='blue', label='L(g)')
    
    # Etiquetas y título
    plt.xlabel('k')
    plt.ylabel('L(g)')
    plt.title('Gráfica de L(g) vs k')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Mostrar gráfica
    plt.show()
    
def grafica_ls_vs_knns(df):
    df_k = pd.DataFrame(df[[i for i in df.columns if i[:6] == 'mean-k']])
    fig, ax = plt.subplots(figsize=(8,5))

    # Graficar cada columna en el mismo eje con diferente color
    for col in df_k.columns:
        ax.plot(df_k.index, df_k[col], marker='o', label=col)
    
    # Etiquetas y estilo
    ax.set_xlabel("num_datos")
    ax.set_ylabel("Valor")
    ax.set_title("Comparación de mean-k-NN para distintos k")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    
def plot_decision_boundary(model, X, y, title):
    # Crear nueva figura cada vez
    plt.figure(figsize=(6, 5))
    
    # Crear grid en el espacio 2D
    x_min, x_max = X['x1'].min() - 1, X['x1'].max() + 1
    y_min, y_max = X['x2'].min() - 1, X['x2'].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Predicciones sobre el grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Graficar fronteras y puntos
    cmap = ListedColormap(['red', 'blue'])
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    plt.scatter(X['x1'], X['x2'], c=y,cmap=cmap,edgecolor="k", s=40)
    plt.title(title)
    plt.show()
    
    
def grafica_puntos_y_aprendizaje_qda(n0, n1, tipo_clases, casos_norm, nombre = 'Titulo'):
    # Gráfico de puntos 

    caso = tipo_clases
    pi0 = n0/(n0+n1)
    pi1 = 1- pi0
    df, mu0, mu1, sigma0, sigma1 = simulacion(casos_norm,caso, n0, n1) 
    #Grafica los puntos
    grafica_puntos(df , nombre)

    qda = QuadraticDiscriminantAnalysis(priors=[pi0, pi1])
    qda.fit(df[['x1', 'x2']], df['cat'])
    
    plot_decision_boundary(qda, df[['x1', 'x2']], df['cat'], 'Aprendizaje QDA')
    
def grafica_puntos_y_aprendizaje_nb(n0, n1, tipo_clases, casos_norm, nombre = 'Titulo'):
    # Gráfico de puntos 

    caso = tipo_clases
    pi0 = n0/(n0+n1)
    pi1 = 1- pi0
    df, mu0, mu1, sigma0, sigma1 = simulacion(casos_norm,caso, n0, n1) 
    #Grafica los puntos
    grafica_puntos(df , nombre)

    nb = GaussianNB(priors=[pi0, pi1])
    nb.fit(df[['x1', 'x2']], df['cat'])
    
    plot_decision_boundary(nb, df[['x1', 'x2']], df['cat'], 'Aprendizaje NB')
    
def grafica_puntos_y_aprendizaje_knn(n0, n1, tipo_clases, casos_norm, nombre = 'Titulo'):
    # Gráfico de puntos 

    caso = tipo_clases
    pi0 = n0/(n0+n1)
    pi1 = 1- pi0
    df, mu0, mu1, sigma0, sigma1 = simulacion(casos_norm,caso, n0, n1) 
    #Grafica los puntos
    grafica_puntos(df , nombre)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(df[['x1', 'x2']], df['cat'])
    
    plot_decision_boundary(knn, df[['x1', 'x2']], df['cat'], 'Aprendizaje NB')
    
def heatmapdiff(df):

    cols = [c for c in df.columns if c != "mean-bayes"]

    # Calculamos las diferencias respecto a mean_bayes
    df_diff = df[cols].subtract(df["mean-bayes"], axis=0)
    
    # Graficar heatmap
    plt.figure(figsize=(10,6))
    sns.heatmap(df_diff, annot=True, cmap="coolwarm", center=0, cbar_kws={'label': 'Diferencia'})
    plt.title("Diferencia de cada modelo respecto al error de bayes")
    plt.xlabel("Modelos")
    plt.ylabel("Tamaño Pob 1")
    plt.tight_layout()
    plt.show()
            
    
def grafica_bandas_variabilidad(df_final, df_final_mean):
    """
    Gráfico de líneas con bandas que muestran la variabilidad (±1 desviación)
    """
    plt.figure(figsize=(14, 8))
    
    # Modelos a destacar (puedes ajustar esta lista)
    modelos_destacados = ['Naive Bayes', 'LDA', 'QDA', 'bayes', 'k-NN (k=1)', 'k-NN (k=5)']
    
    for modelo in modelos_destacados:
        mean_col = f'mean-{modelo}'
        dvs_col = f'dvs-{modelo}'
        
        if mean_col in df_final_mean.columns and dvs_col in df_final.columns:
            means = df_final_mean[mean_col]
            dvs = df_final[dvs_col]
            
            # Línea principal
            line = plt.plot(means.index, means, 
                           label=modelo, linewidth=2, marker='o')
            color = line[0].get_color()
            
            # Banda de variabilidad
            plt.fill_between(means.index, 
                           means - dvs, 
                           means + dvs, 
                           alpha=0.2, color=color)
    
    plt.xlabel('Tamaño Población Clase 1 (n1)')
    plt.ylabel('Error de Clasificación L(g)')
    plt.title('Evolución del Error con Bandas de Variabilidad (±1 Desviación Estándar)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

            

        
        
        
        
        