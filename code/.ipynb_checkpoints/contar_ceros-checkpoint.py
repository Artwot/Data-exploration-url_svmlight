import findspark
findspark.init()
import numpy as np
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf

conf = (SparkConf()
        .setAppName("Calcular Ceros") \
        .set('spark.driver.cores', '3') \
        .set('spark.executor.cores', '3') \
        .set('spark.driver.memory', '2G') \
        .set('spark.sql.autoBroadcastJoinThreshold', '-1') \
        .set('spark.executor.memory', '3G'))
sc = SparkContext(conf=conf)

spark = SparkSession.builder.getOrCreate()

cont = 0
array_cont_filas = []

rdd_data = spark.read.format("libsvm")\
    .load("../data/url_svmlight/403995_x_instances_10.svm")

rdd_data.printSchema()

def contar_ceros(fila):
    global cont 
    global array_cont_filas
    #for i in range(len(fila[1])):
    for i in range(len(fila)):
        if(fila[i] == 0.0):
            cont += 1
    array_cont_filas.append(cont)
    cont = 0

rdd_data.foreach(contar_ceros)

print(array_cont_filas)

array = [0.0,
1547045,
0.0,
1547044,
1547042,
0.0,
1547053,
1547040,
1547050,
1547054,
1547044,
1547002,
0.0,
1547036,
1547050,
1547050,
0.0,
1547011,
1547040,
0.0]

contar_ceros(array)

suma = 0 
for i in range(len(array) - 1):
    suma += array[i] + array[i + 1]
print(suma)


