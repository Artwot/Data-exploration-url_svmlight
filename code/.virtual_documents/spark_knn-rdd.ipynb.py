import findspark
findspark.init()
from pyspark.sql import SparkSession
import random
import math
import sys
import numpy as np 


# Build the SparkSession
spark = SparkSession.builder \
    .master("local[6]") \
    .appName("Data exploration URL - KNN Spark RDD") \
    .config("spark.executor.memory", "4gb") \
    .getOrCreate()

sc = spark.sparkContext


sc._conf.getAll()


sc


def euclidean_distance(row1, row2):
    distance = 0.0
    for column in range(len(row1[1])):
        distance += pow(row1[1][column] - row2[1][column], 2)
    distance = math.sqrt(distance)
    return distance


def get_neighbors(train, test_row, k):
    distances = []
    total_train_rows = len(train)
    for train_row in range(total_train_rows):
        distance = euclidean_distance(test_row, train[train_row])
        if(distance > 0.0):
           distances.append((train[train_row], distance))
    distances.sort(key = lambda tup: tup[1])
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors


def predict_classification(train, test_row, k):
    neighbors = get_neighbors(train, test_row, k)
    output_labels = [row[0] for row in neighbors]
    prediction = max(set(output_labels), key=output_labels.count)
    return prediction


def accuracy(real_labels, predicted):
    correct = 0
    total_rows = len(real_labels)
    for i in range(total_rows):
        if(real_labels[0][0] == predicted[i - 1]):
            correct += 1
    print("Correct labels: ", correct, 'of', (total_rows))
    accuracy = correct / float(total_rows)
    return accuracy


def k_nearest_neighbors(train, test, k):
    predictions = []
    total_test_rows = len(test)
    for test_row in range(total_test_rows):
        output = predict_classification(train, test[test_row], k)
        predictions.append(output)
    mean_accuracy = accuracy(test, predictions)
    print("Mean accuracy: " + str(mean_accuracy))


# Load training data
data = spark.read.format("libsvm")\
    .load("../data/url_svmlight/Dimension_100_x_500000.svm")
# Split the data into train and test
seed = random.randrange(500, 1300, 2)
splits = data.randomSplit([0.7, 0.3], 1234)

train = splits[0]
test = splits[1]


#k_nearest_neighbors(train, test, k = 1)


# Prueba de la función euclidean_distance(), se mandan dos renglones del dataset total
length = data.count()               # Se obtiene el total de renglones en el dataset
start_time = time.time()
for row in range(1, (length + 1)):
    distance = euclidean_distance(data.head(1)[-1], data.head(row)[-1])
    #print("Dinstancia del renglon 1 con el", row, ":", distance)
end_time = time.time()
print(tiempo(start_time, end_time))


# Prueba de la función get_neighbors(), se envían como args. el dataframe, un renglón de este y el número de vecinos.
"""
start_time = time.time()
length = data.count() + 1
neighbors = get_neighbors(data, data.head(1)[-1], k=3)
for neighbor in neighbors:
    print(neighbor)
end_time = time.time()
print(tiempo(start_time, end_time))
"""


# Prueba de la función get_neighbors(), se envían como args. el dataframe, un renglón de este y el número de vecinos y n es igual al número de renglón.
n = 1
print("Row class:", data.head(n)[-1][0] ) # CLase/Label
prediction = predict_classification(data, data.head(n)[-1], k=3)
print('Expected label: get_ipython().run_line_magic("d,", " Got: %d.' % (data.head(n)[-1][0], prediction))")


# Se asignan los RDD para el posterior procesamiento
rdd_train = train.rdd
rdd_test = test.rdd
rdd_total = data.rdd


# Se utilizan dos array de numpy para alamacenar las instancias del set de entrenamiento y procesar con un RDD y el segundo array almacena las etiquetas. 
#train_array = np.array(train.select('features').collect(), dtype=float)
#train_array_labels = np.array(train.select('label').collect(), dtype=float)


# Etiquteas de las instancias del conjunto de test. 
train_array = np.array(train.collect(), dtype=object)
test_array = np.array(test.collect(), dtype=object)
total_array = np.array(data.collect(), dtype=object)


total_array[0]


start_time = time.time()
length = len(total_array)
for row in range(length):
    distance = euclidean_distance(total_array[0], total_array[row])
    #print("Dinstancia del renglon 1 con el", row, ":", distance)    
end_time = time.time()
print(tiempo(start_time,end_time))


# Prueba de la función get_neighbors(), se envían como args. el dataframe, un renglón de este y el número de vecinos.
start_time = time.time()
length = data.count() + 1
neighbors = get_neighbors(total_array, total_array[0], k=3)
for neighbor in neighbors:
    print(neighbor)
end_time = time.time()
print(tiempo(start_time, end_time))


n = 1
print("Row class:", data.head(n)[-1][0] ) # CLase/Label
prediction = predict_classification(total_array, total_array[0], k=3)
print('Expected label: get_ipython().run_line_magic("d,", " Got: %d.' % (total_array[0][0], prediction))")


print(test_array[0][0])


start_time = time.time()
k_nearest_neighbors(train_array, test_array, k = 3)
end_time = time.time()
print(tiempo(start_time, end_time))


print('RDD de entrenamiento: ' + str(rdd_train.count()))
print('RDD de test: ' + str(rdd_test.count()))


# Metodo que guarda cada renglon en un archivo .svm
def save_file(data):
    file = open('../data/url_svmlight/Distancia_euclideana_100_x_500000.svm', 'a')
    file.write(data)
    file.close()


"""
[summary:
    Método para calcular la distancia euclídea entre cada una de las 
    columnas del conjunto de test respecto a las columnas del conjunto
    de entrenamiento.
]

Args:
    instance ([pyspark.sql.types.Row]): [
        Recibe cada una de las instancias que hay en el dataset
    ]
"""
def euclidean_distance(instance):
    distance = 0
    instance_distance = ''
    for row in range(len(train_array)):
        instance_distance += str(train_array_labels[row][0]) + ' '
        for column in range(len(instance[1])):
            distance = pow(train_array[row][0][column] - instance.features[column], 2)
            distance = math.sqrt(distance)
            # instance_distance += str(column + 1) +':' + str(distance) + ' ' # -> Si quisiera poner los indices de cada caracteristica.
            instance_distance += str(distance) + ' '
        instance_distance += '\n'
    save_file(instance_distance)


# Ejecuta el método que calcula la distancia euclídea entre los puntos euclidean_distance()
test.foreach(euclidean_distance)


rdd_samp1 = sc.textFile('../data/url_svmlight/arch_prb.svm')
rdd_samp2 = sc.textFile('../data/url_svmlight/arch_prb1.svm')
rdd_samp3 = sc.textFile('../data/url_svmlight/arch_prb2.svm')


five_nearest1 = rdd_samp1.takeOrdered(5)
five_nearest2 = rdd_samp2.takeOrdered(5)
five_nearest3 = rdd_samp3.takeOrdered(5)


def class_average(five_nearest):
    mean = 0
    for i in range(5):
        mean += float(five_nearest[i][0])
    mean = mean / 5
    if(mean > 0.5):
        print('Clase K-NN: 1')
        return 1
    else:
        print('Clase K-NN: 0')
        return 0


def accuracy():
    lista = [five_nearest1, five_nearest2, five_nearest3]
    accuracy = 0.0
    for i in range(len(test_array_labels)):
        if(test_array_labels[i][0] == class_average(lista[i])):
            accuracy += 1
        print('Clase Real: ' + str(test_array_labels[i][0]))
        print('\n')
    accuracy = accuracy / len(test_array_labels)
    print('Accuracy: ' + str(accuracy))


accuracy()


lista = [five_nearest1, five_nearest2, five_nearest3]


# listaclass_average


# five_nearest[0][0] # Clase


# rdd_prueba2 = sc.textFile('../data/url_svmlight/Distancia_euclideana_5_x_76.svm')


five_nearest2 = rdd_prueba.takeOrdered(5)
type(five_nearest2)


five_nearest2[0][0]


import time
def tiempo(start, end):
    print("Tiempo de ejecución: ", (end - start))


# Prueba de la función euclidean_distance(), se mandan dos renglones del dataset total

start  = time.time()
length = data.count()               # Se obtiene el total de renglones en el dataset
for row in range(1, (length + 1)):
    distance = euclidean_distance(data.head(1)[-1], data.head(row)[-1])
    #print("Dinstancia del renglon 1 con el", row, ":", distance)

end = time.time()

tiempo(start, end)


start = time.time()
data.head(175)[-1]
end = time.time()

tiempo(start, end)


data.select('* where label=0')


data.head(1)[-1][1][0]


for i in range(76):
    print(data.head(10)[-1][1][i])
    #print(i)



