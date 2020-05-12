# CS205 final_project
## Project Statement
Scene classification is one of the fundamental problem in the area of computer vision. However, the databases that were studied in this area have covered a limited range of categories and do not provide diverse scenes. Despite the massive size of the datasets currently used, the number of scene categories that the largest available dataset contains is fewer than 20 classes. Thus, in this project we are interested in training a Convolutional Neural Network on a large-scale dataset with a variety of categories (397 classes).

Convolutional Neural Networks (CNN) are widely used in image analysis. CNN is usually quite computationally demanding to train, even more intensive if the training dataset is larger and hyper-parameter tuning is required.  In this project, we aim to explore parallel CNN training for huge image dataset, find feasible solution and compare runtime and performance in different settings. More specifically, we will use a huge image dataset and train CNN for label classification with Spark and using parallelization framework available on AWS. 

For detailed report, please visit the website [here](https://sites.google.com/view/cs205finalproject-parallelcnn/home)

## Environment Builing and Infrastructure Setup

We have separate code for running our model in a local machine or on a AWS cluster. I will walk through them one by one

### Getting Data:
You can find the Sun397 Data [here](https://vision.cs.princeton.edu/projects/2010/SUN/). Downlaod the Sun397.tar.gz to your local disk and unzip it. If you want to train the model on cluster, please upload them to the S3 bucket.

### Local Machine Setup:
You need to use conda and pip to set up a new envrionment for this application

To create a new environment with `conda create --name env_name python=3.7`

Activate your environment with `conda activate env_name`

Clone the repository with `git clone https://github.com/CS205group11/final_project.git` and enter the final project directory

Install all dependencies with `pip install -r requirements.txt`

Edit the `cnn_spark_local.py` and change the `datapath` variable to the Sun397 folder in your local machine

You can then train by yourself by type the command `spark-submit cnn_spark_loacl.py` and watch the training process

If you encounter memory issue. You may need to add `--driver-memory= Size` where `Size` is the maxium memory needed



