#!/usr/bin/env python
# coding: utf-8

# # Clasificador K-NN en Spark usando pyspark.RDD

# ### Se importan las librerías necesarias

# In[12]:


import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.ml.feature import MaxAbsScaler

import random
import math
import time
import numpy as np 
import sys 
import os

path = str(sys.argv[1])
k_nn = int(sys.argv[2])


# ### Se crea la sesión y config. de Spark

# In[2]:


conf = (SparkConf()
        .setAppName("Data exploration URL - KNN Spark RDD") \
        .set('spark.driver.cores', '6') \
        .set('spark.executor.cores', '6') \
        .set('spark.driver.memory', '6G') \
        .set('spark.master', 'local[6]') \
        .set('spark.sql.autoBroadcastJoinThreshold', '-1') \
        .set('spark.executor.memory', '6G'))
sc = SparkContext(conf=conf)


# In[3]:


spark = SparkSession.builder.getOrCreate()


# In[4]:


sc._conf.getAll()


# In[5]:


sc


# ### Función para calcular el tiempo de ejecución

# In[6]:


def tiempo(start, end):
    medida = 'segundos'
    tiempo = end - start
    if (tiempo >= 60):
        tiempo = tiempo / 60
        medida = 'minutos'
    else:
        if (tiempo >= 3600):
            tiempo = tiempo / 3600
            medida = 'horas'
    print("Tiempo de ejecución: ", round(tiempo, 2), medida)


# ### Calcular la distancia euclideana.
# #### Summary:
#         Se calcula la distancia entre las columnas de dos renglones de un dataset, funciona
#         con argumentos provenientes de un renglón de un dataframe de Spark.
# #### Args: 
#         row1(numpy.ndarray): Recibe una instancia del dataset
#         row2(pyspark.ml.linalg.SparseVector): Recibe una instancia del dataset

# In[7]:


def euclidean_distance(row1, row2):
    distance = 0.0
    columns = len(row1[0])
    for column in range(columns):
        distance += pow(row1[0][column] - row2[column], 2)
    distance = math.sqrt(distance)
    return distance


# ### Obtener los vecinos más cercanos.
# #### Summary: 
#       Se recorre cada renglón del dataframe dado y se calcula la distancia entre cada 
#       uno de estos y el renglón de prueba.
#       El RDD "distances", almacenará las distancias calculadas, 
#       posteriormente se ordena de modo ascendente y se almancenan los primeros k-elementos 
#       en la lista "k_neighbors"
# 
# #### Args: 
#       train(pyspark.rdd.RDD): Recibe el conjunto de entrenamiento
#       test_row(numpy.ndarray): Recibe una instancia del conjunto de test
#       k(int): Número de vecinos que se desean obtener

# In[8]:


def get_neighbors(train, test_row, k):
    rdd_distances = train.map(lambda element: (element[0], euclidean_distance(test_row, element[1])))
    rdd_distances = rdd_distances.filter(lambda element: element[1] > 0.0)
    rdd_distances.repartition(50)
    k_neighbors = rdd_distances.takeOrdered(k, key= lambda  x: x[1]) 
    return k_neighbors


# ### Predecir las etiquetas usando k-nn.
# #### Summary:
#       Se obtiene la lista de los k-vecinos más cercanos, y se almacena el valor de
#       la etiqueta en la lista "output_labels". Posteriormente se calcula el valor 
#       promedio de las etiquetas y se almacena en la variable "prediction" y se retorna.
# 
# #### Args: 
#       train(pyspark.rdd.RDD): Recibe el conjunto de entrenamiento
#       test_row(numpy.ndarray): Recibe una instancia del conjunto de test
#       k(int): Número de vecinos que se desean obtener

# In[9]:


def predict_classification(train, test_row, k):
    neighbors = get_neighbors(train, test_row, k)
    output_labels = [row[0] for row in neighbors]
    prediction = max(set(output_labels), key=output_labels.count)
    return prediction


# ### Clacular el porcentaje de exactitud.
# #### Summary:
#       Esta función calcula el porcentaje de exactitud del uso de k-NN, comparando
#       las etiquetas reales de las instancias del dataset de entrenamiento y las
#       etiquetas obtenidas mediante la predicción usando k-NN.
# #### Args: 
#       real_labels(numpy.ndarray): Recibe el dataframe de test que contiene los
#                                                     valores reales de las etiquetas
#       predicted(list): Lista con las etiquetas obtenidas mediante K-NN

# In[10]:


def accuracy(real_labels, predicted):
    correct = 0
    total_rows = len(real_labels)
    for i in range(total_rows):
        if(real_labels[i] == predicted[i]):
            correct += 1
    print("Correct labels: ", correct, 'of', (total_rows))
    accuracy = correct / float(total_rows)
    return accuracy


# ### Crear la función que calcule los vecinos más cercanos.
# #### Summary:
#       Se asignan los parámetros para calcular los k-vecinos más cercanos y hacer predicciones
#       de las etiquetas a las que pertenecen, calculando la distancia entre las columnas de cada
#       uno de los renglones del dataframe de "test" y el de "train", comparando las 
#       reales con las otenidas por el clasificador y, finalmente, dado el porcentaje de exactitud obtenido. 
# #### Args: 
#       train(pyspark.rdd.RDD): Recibe el conjunto de entrenamiento
#       test(pyspark.rdd.RDD): Recibe el conjunto de test
#       k(int): Número de vecinos que se desean obtener

# In[11]:


def k_nearest_neighbors(train, test, k):
    predictions = []
    total_test_rows = test.count()
    for index in range(total_test_rows):
        test_row = np.array(test.zipWithIndex().filter(lambda element: element[1] == index).map(lambda element: element[0][1]).collect(), dtype = object)
        output = predict_classification(train, test_row, k)
        predictions.append(output)
    labels_array = np.array(test.map(lambda x: x[0]).collect(), dtype = float)
    mean_accuracy = accuracy(labels_array, predictions)
    print("Mean accuracy: " + str(mean_accuracy))


# ## Se cargan los datos al dataframe 

# In[13]:


# Load training data
data = spark.read.format("libsvm").load(path)


# In[14]:


data.printSchema()


# ### Normalización

# In[15]:


scaler = MaxAbsScaler(inputCol="features", outputCol="features_norm")

# Compute summary statistics and generate MaxAbsScalerModel
scalerModel = scaler.fit(data)

# rescale each feature to range [-1, 1].
scaledData = scalerModel.transform(data)

scaledData = scaledData.drop("features")


# In[16]:


#Dividir los datos en conjunto de train y de test
seed = 1234
splits = scaledData.randomSplit([0.7, 0.3], seed)

train = splits[0]
test = splits[1]

# Se asignan los RDD para el posterior procesamiento
rdd_train = train.rdd
rdd_test = test.rdd
# rdd_total = data.rdd

scaledData.head(1)


# In[17]:


#rdd_test.getNumPartitions()
rdd_train1 = rdd_train.repartition(30)
rdd_test1 = rdd_test.repartition(30)


# In[18]:


rdd_train.getNumPartitions()


# In[19]:


rdd_test.getNumPartitions()


# ## Se invoca al método y se envían los parámetros

# In[20]:


start_time = time.time()
k_nearest_neighbors(rdd_train1, rdd_test1, k = k_nn)
end_time = time.time()
print(tiempo(start_time, end_time))


# # Spark session stop

# In[40]:


sc.stop()

