# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import findspark
findspark.init()

from pyspark.sql import SparkSession
import random
import math
import sys
import numpy as np


# %%
# Build the SparkSession
spark = SparkSession.builder     .master("local[6]")     .appName("Data exploration URL - KNN")     .config("spark.executor.memory", "4gb")     .getOrCreate()

sc = spark.sparkContext


# %%
sc._conf.getAll()


# %%
# Load training data
data = spark.read.format("libsvm")    .load("../data/url_svmlight/Dimension_5_x_76.svm")
# Split the data into train and test
seed = random.randrange(500, 1300, 2)
splits = data.randomSplit([0.7, 0.3], 1234)

train = splits[0]
test = splits[1]


# %%
# Se asignan los RDD para el posterior procesamiento
rdd_train = train.rdd
rdd_test = test.rdd


# %%
train_array = np.array(train.select('features').collect(), dtype=float)
train_array_labels = np.array(train.select('label').collect(), dtype=float)


# %%
train_array_labels[6][0]


# %%
print('RDD de entrenamiento: ' + str(rdd_train.count()))
print('RDD de test: ' + str(rdd_test.count()))


# %%
rdd_test.collect()


# %%
# Metodo que guarda cada renglon en el archivo .svm
def save_file(data, count):
    file = open('../data/url_svmlight/Distancia_euclideana_5_x_76' + str(count) + '.svm', 'a')
    file.write(data)
    file.close()


# %%
def euclidean_distance(instance):
    """[summary:
        Método para calcular la distancia euclídea entre cada una de las 
        columnas del conjunto de test respecto a las columnas del conjunto
        de entrenamiento.
    ]

    Args:
        instance ([pyspark.sql.types.Row]): [
            Se recibe cada una de las instancias que hay en el dataset
        ]
    """
    distance = 0
    instance_distance = ''
    print('=================' + str(instance.label))
    for row in range(len(train_array)):
        instance_distance += str(train_array_labels[row][0]) + ' ' 
        # print('Columna ' + str(column) + ': ' + str(instance.features[column]))
        # print(test_array[0][0][column])
        for column in range(len(instance[1])):
            # print('+++++++++++++++++++++++++++++++++++++++++++++')
            # print('Instancia del train : ' + str(row))
            # print(str(instance.features[column]))
            # print('Columna: ' + str(column))
            # print(train_array[row][0][column])
            distance = pow(train_array[row][0][column] - instance.features[column], 2)      
            # print('Distancia euclideana: ')
            distance = math.sqrt(distance)
            # print(str(distance))
            instance_distance += str(column + 1) +':' + str(distance) + ' ' # -> Si quisiera poner los indices de cada caracteristica.
            # instance_distance += str(distance) + ' '
        instance_distance += '\n'
    save_file(instance_distance)


# %%
# Ejecuta el método que calcula la distancia euclídea entre los puntos
test.foreach(euclidean_distance)


# %%
rdd_prueba = sc.textFile('../data/url_svmlight/Distancia_euclideana_5_x_76_2.svm', 3)


# %%
def mean(iterator):
    for instance in iterator:
        lista = instance.split(' ')
        sort_list = sorted(lista)
        print(sort_list)
    print('================================= FIN ==========================')


# %%
# rdd4 = rdd_prueba.zipWithIndex()
# rdd4.getNumPartitions()
# rdd4.collect()


# %%
r = rdd_prueba.foreachPartition(mean).takeOrdered(5)


# %%
# five_nearest[0][0] # Clase


# %%
def class_average():
    for i in range(len(five_nearest)):
        


# %%
# rdd_prueba2 = sc.textFile('../data/url_svmlight/Distancia_euclideana_5_x_76.svm')


# %%
five_nearest2 = rdd_prueba.takeOrdered(5)
type(five_nearest2)


# %%



