import pandas as pd
#La funcion recibe el nombre del archivo a extraer y dos booleanos, months y days, que indican si queremos
#extraer ambos datos de la base de datos, en caso negativo los quita antes de regresarla
def extraerBD(nombre = "bank-full.csv", month = True, day = True):
    X = pd.read_csv(nombre, delimiter=';')

    if not month and "month" in X.columns:
        X = X.drop(['month'], axis = 1) #<- Por si queremos ignorar el mes
    if not day and "day" in X.columns:
        X = X.drop(['day'], axis = 1) #<- Por si queremos ignorar el dia del mes

    Y = X['y'] #<- Una df con todas las variables de respuesta
    Y = Y.map({"no": 0, "yes": 1}) # Codificamos no a 0 y yes a 1

    X = X.drop(['y'], axis = 1)#Quitamos de X la columna y
    X_dummies = pd.get_dummies(X, drop_first=False).astype(int)#Creamos una nueva df con las variables categoricas como dummies

    return X, X_dummies, Y
