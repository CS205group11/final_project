import tensorflow_datasets as tfds
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
from tensorflow.keras.preprocessing import image
import os
from pyspark import SparkConf, SparkContext
import string
import sys
import re

def load_data(path,sun397path='/Users/zdd/Desktop/SUN397'):
    target_class=path[3:-25]
    target_class=output_dict[target_class]
    full_path=sun397path+path
    img=image.load_img(full_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array=img_array/255
    return (img_array,target_class)

output_dict={}
with open ('/Users/zdd/Desktop/SUN397/ClassName.txt') as f:
    current_class=0
    for lines in f.read().splitlines():
        name=lines[3:]
        output_dict[name]=current_class
        current_class=current_class+1
conf = SparkConf().setMaster('local[4]').setAppName('read_data')
sc = SparkContext(conf = conf)
RDD=sc.textFile('./Partitions/Training_02.txt')
result=RDD.map(lambda line:load_data(line) )
result.saveAsTextFile("tmpoutput.txt")

