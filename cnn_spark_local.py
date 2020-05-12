# Load useful libraries
import numpy as np
import pandas as pd
import time
# keras utility
from get_model import get_model
from keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
# import spark and elephas
from pyspark import SparkConf, SparkContext
from elephas.spark_model import SparkModel
#data path to sun397, user should chaneg that to your own path
data_path='/Users/zdd/Desktop/SUN397'
#code to get a dictionary for label
output_dict={}
with open (data_path+'/ClassName.txt') as f:
current_class=0
for lines in f.read().splitlines():
    name=lines[3:]
    output_dict[name]=current_class
    current_class=current_class+1
#custom function to process iamage and get label
def load_data(path,sun397path=data_path):
    target_class=path[3:-25]
    target_class=output_dict[target_class]
    target_class=to_categorical(target_class,397)
    full_path=sun397path+path
    img=image.load_img(full_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array=img_array/255
    return (img_array,target_class)

#data processing with spark
conf = SparkConf().setMaster('local[2]').setAppName('cnn_local')
sc = SparkContext(conf = conf)
sc.setLogLevel("WARN")
RDD_train=sc.textFile('./Training_01.txt')
RDD_test==sc.textFile('./Testing_01.txt')
train_data=RDD_train.map(lambda line:load_data(line) )
test_data=RDD_test.map(lambda line:load_data(line) )
#Model Training
model=get_model(32)
spark_model = SparkModel(model, frequency='epoch', mode='synchronous')
spark_model.fit(train_data, epochs=5, batch_size=32, verbose=1, validation_split=0.1)
#Model Evaluating
x_test=np.array(test_data.map(lambda tuple:tuple[0]).collect())
y_test=np.array(test_data.map(lambda tuple:tuple[1]).collect())
accuracy=spark_model._master_network.evaluate(x_test,y_test)[1]
print('Accuracy is {}'.format(accuracy))


