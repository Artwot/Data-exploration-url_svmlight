# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


pwd


# Importar el data set
dataset = pd.read_csv('../data/sklearn_format/url_10x76.csv', header=None)

X = dataset.iloc[:, 0:len(dataset.columns) - 2].values
y = dataset.iloc[:, -1].values


dataset.head(5)


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Escalado de variables print(Me parece que esta celda no es muy necesaria, pues los datos de la base están en un rango de 0-1)
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)


# Shapes
print('Shape de X_train: ' + str(X_train.shape))
print('Shape de X_test: ' + str(X_test.shape))
print('Shape de Y_train: ' + str(y_train.shape))
print('Shape de y_test: ' + str(y_test.shape))


# Ajustar el clasificador en el Conjunto de entrenamiento
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
print("of K-NN classifier on training set: ", classifier.score(X_train, y_train))
print("of K-NN classifier on test set: ", classifier.score(X_test, y_test))


# Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)


# Importar las métricas de sklearn para calcular la exactitud
from sklearn import metrics
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))


# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))




