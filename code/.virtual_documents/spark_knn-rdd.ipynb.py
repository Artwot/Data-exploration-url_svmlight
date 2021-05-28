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

#path = str(sys.argv[1])
#atr = int(sys.argv[2])


conf = (SparkConf()
        .setAppName("Data exploration URL - KNN Spark RDD") \
        .set('spark.driver.cores', '6') \
        .set('spark.executor.cores', '6') \
        .set('spark.driver.memory', '6G') \
        .set('spark.master', 'local[6]') \
        .set('spark.sql.autoBroadcastJoinThreshold', '-1') \
        .set('spark.executor.memory', '6G'))
sc = SparkContext(conf=conf)


spark = SparkSession.builder.getOrCreate()


sc._conf.getAll()


sc


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


def euclidean_distance(row1, row2):
    distance = 0.0
    columns = len(row1[0])
    for column in range(columns):
        distance += pow(row1[0][column] - row2[column], 2)
    distance = math.sqrt(distance)
    return distance


def get_neighbors(train, test_row, k):
    rdd_distances = train.map(lambda element: (element[0], euclidean_distance(test_row, element[1])))
    rdd_distances = rdd_distances.filter(lambda element: element[1] > 0.0)
    rdd_distances.repartition(50)
    k_neighbors = rdd_distances.takeOrdered(k, key= lambda  x: x[1]) 
    return k_neighbors


def predict_classification(train, test_row, k):
    neighbors = get_neighbors(train, test_row, k)
    output_labels = [row[0] for row in neighbors]
    prediction = max(set(output_labels), key=output_labels.count)
    return prediction


def accuracy(real_labels, predicted):
    correct = 0
    total_rows = len(real_labels)
    for i in range(total_rows):
        if(real_labels[i] == predicted[i]):
            correct += 1
    print("Correct labels: ", correct, 'of', (total_rows))
    accuracy = correct / float(total_rows)
    return accuracy


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


# Load training data
data = spark.read.format("libsvm")\
    .option("header", "false")\
    .option("inferSchema","true")\
    .load("../data/url_svmlight/807990_x_instances_10.svm")
    # .load("/home/jsarabia/Documents/IA/datasets/Data-exploration/807990_x_instances_30.svm")
    # .load("../data/url_svmlight/Dimension_100_x_1000.svm")


data.printSchema()


scaler = MaxAbsScaler(inputCol="features", outputCol="features_norm")

# Compute summary statistics and generate MaxAbsScalerModel
scalerModel = scaler.fit(data)

# rescale each feature to range [-1, 1].
scaledData = scalerModel.transform(data)

scaledData = scaledData.drop("features")


#Dividir los datos en conjunto de train y de test
seed = 1234
splits = scaledData.randomSplit([0.6, 0.4], seed)

train = splits[0]
test = splits[1]

# Se asignan los RDD para el posterior procesamiento
rdd_train = train.rdd
rdd_test = test.rdd
# rdd_total = data.rdd

scaledData.head(1)


#rdd_test.getNumPartitions()
rdd_train1 = rdd_train.repartition(30)
rdd_test1 = rdd_test.repartition(30)


rdd_train1.getNumPartitions()


rdd_test1.getNumPartitions()


start_time = time.time()
k_nearest_neighbors(rdd_train1, rdd_test1, k = 7)
end_time = time.time()
print(tiempo(start_time, end_time))


sc.stop()


# Se asignan los RDD para el posterior procesamiento
rdd_train = train.rdd
rdd_test = test.rdd
rdd_total = data.rdd


# Se agrega un índice a las instancias para poder recorrerlas posteriormente mediante un filtro.
rdd_index = rdd_total.zipWithIndex()
# Se selecciona solo la columna que contiene los valores de las características.
rdd_columns = rdd_total.map(lambda x: x[1])


# Se prueba el método de distancia euclideana con RDD
# Renglón no. 1
rdd_row1 = rdd_index.filter(lambda x: x[1] == 0)
# Se transforma en un array de Numpy solo con los valores de las columnas
row1 = np.array(rdd_row1.map(lambda element: element[0][1]).collect(), dtype = object)
# Las distancias se almacenan en un RDD
rdd_distances = rdd_columns.map(lambda x: euclidean_distance(row1, x))
start_time = time.time()
rdd_distances.collect()
end_time = time.time()
print(tiempo(start_time,end_time))


# Prueba de la función get_neighbors()
# Renglón no. 1
rdd_row1 = rdd_index.filter(lambda x: x[1] == 0)
# Se transforma en un array de Numpy solo con los valores de las columnas
row1 = np.array(rdd_row1.map(lambda element: element[0][1]).collect(), dtype = object)
start_time = time.time()
print(get_neighbors(rdd_total, row1, k = 5))
end_time = time.time()
print(tiempo(start_time,end_time))


	20 s# Prueba de la función predict_classification()
# Renglón no. 1
rdd_row1 = rdd_index.filter(lambda x: x[1] == 0)
# Se transforma en un array de Numpy solo con los valores de las columnas
row1 = np.array(rdd_row1.map(lambda element: element[0][1]).collect(), dtype = object)
start_time = time.time()
prediction = predict_classification(rdd_total, row1, 3)
print('Expected label: get_ipython().run_line_magic("d,", " Got: %d.' % (rdd_row1.take(1)[0][0][0], prediction))")
end_time = time.time()
print(tiempo(start_time,end_time))


sc.stop()
