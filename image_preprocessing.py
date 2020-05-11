# Load useful libraries
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import initializers
from tensorflow.keras.metrics import *

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
plt.gray()

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from pyspark import SparkConf, SparkContext
from pyspark.sql.session import SparkSession
import glob

with open("SUN397/ClassName.txt", "r") as f:
    paths = f.read().split('\n')

output_dict = {}
with open("SUN397/ClassName.txt", "r") as f:
    current_class=0
    for lines in f.read().splitlines():
        name=lines[3:]
        output_dict[name]=current_class
        current_class=current_class+1

def load_transform_data(path,sun397path='./'):
    target_class=path[9: -25]
    target_class=output_dict[target_class]
    full_path=sun397path+path
    img=load_img(full_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array=img_array/255
    return (img_array,target_class)

all_img = []
for path in paths:
    img_path = 'SUN397' + path + '/*.jpg'
    all_img += glob.glob(img_path)

all_img = np.array(all_img)
img_label = np.array([output_dict[img[9:-25]] for img in all_img])

with open("img_paths.txt","w") as f:
    for row in all_img:
        print(row, file = f)

conf = SparkConf().setMaster('local').setAppName('P22')
sc = SparkContext(conf = conf)
spark = SparkSession(sc)
RDD=sc.textFile('img_paths_sub.txt')
result=RDD.map(lambda line:load_transform_data(line) )
result.saveAsTextFile("tmpoutput.txt")