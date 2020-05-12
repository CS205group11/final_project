from elephas.spark_model import SparkModel
# Load useful libraries                                                         
import numpy as np
import pandas as pd
from get_model import get_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
#import spark library
from pyspark import SparkConf, SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark.sql.types import ArrayType
from pyspark.ml.image import ImageSchema
conf = SparkConf().setAppName('project')
sc = SparkContext(conf = conf)
spark = SparkSession(sc)
#read in image as spark dataframe
df = spark.read.format("image").load("s3a://sun397-new/")
# function to convert binary image data to numpy array
def to_array(image):
    height = image.height
    width = image.width
    nChannels = image.nChannels
    return np.ndarray(
        shape=(height, width, nChannels),
        dtype=np.uint8,
        buffer=image.data,
        strides=(width * nChannels, nChannels, 1))
#function to convert label to one-hot encofing
def convert_one_hot(labels):
    one_hot_encoding=np.zeros(397)
    one_hot_encoding[labels]=1
    return one_hot_encoding
    
#train test split
train_df, test_df =df.randomSplit([0.8, 0.2])

#transform spark dataframe to rdd format
train_data=train_df.rdd.map(lambda row:(to_array(row['image']),convert_one_hot(row['Label'])))
test_data=test_df.rdd.map(lambda row:(to_array(row['image']),convert_one_hot(row['Label'])))

#Training part
model=get_model(32)
spark_model = SparkModel(model, frequency='batch', mode='synchronous')
spark_model.fit(train_data, epochs=5, batch_size=32, verbose=1, validation_split=0.1)
#Model_Evaluating
x_test=np.array(test_data.map(lambda tuple:tuple[0]).collect())
y_test=np.array(test_data.map(lambda tuple:tuple[1]).collect())
accuracy=spark_model._master_network.evaluate(x_test,y_test)[1]
print('Accuracy is {}'.format(accuracy))
