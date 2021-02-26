# Import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.sql import SparkSession
import random
import sys
path = str(sys.argv[1])
atr = int(sys.argv[2])
# Build the SparkSession
spark = SparkSession.builder \
    .master("local[3]") \
    .appName("Linear Regression Model") \
    .config("spark.executor.memory", "4gb") \
    .getOrCreate()

sc = spark.sparkContext


# Load training data
data = spark.read.format("libsvm")\
    .load(path)
# Split the data into train and test
seed = random.randrange(500, 1300, 2)
splits = data.randomSplit([0.6, 0.4], seed)

train = splits[0]
test = splits[1]

# specify layers for the neural network:
# input layer of size 4 (features), two intermediate of size 5 and 4
# and output of size 3 (classes)
layers = [atr, 5, 6, 10]

# create the trainer and set its parameters
trainer = MultilayerPerceptronClassifier(
    maxIter=100000, layers=layers, blockSize=100, seed=seed)

# train the model
model = trainer.fit(train)

# compute accuracy on the test set
result = model.transform(test)
predictionAndLabels = result.select("prediction", "label")

evaluator = MulticlassClassificationEvaluator(metricName="f1")
print("f1: " + str(evaluator.evaluate(predictionAndLabels)))
evaluator = MulticlassClassificationEvaluator(metricName="weightedPrecision")
print("weightedPrecision: " + str(evaluator.evaluate(predictionAndLabels)))
evaluator = MulticlassClassificationEvaluator(metricName="weightedRecall")
print("weightedRecall: " + str(evaluator.evaluate(predictionAndLabels)))
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Accuracy: " + str(evaluator.evaluate(predictionAndLabels)))

# python net.py 'ruta' atributos
# python net.py /home/manuel/Escritorio/Proyecto/KDD.dat/part-00000 41
# python net.py '/home/manuel/Escritorio/part-00000' 41
# python net.py /home/manuel/Escritorio/part-00000 41
