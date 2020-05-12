# CS205 final_project
## Project Statement
Scene classification is one of the fundamental problem in the area of computer vision. However, the databases that were studied in this area have covered a limited range of categories and do not provide diverse scenes. Despite the massive size of the datasets currently used, the number of scene categories that the largest available dataset contains is fewer than 20 classes. Thus, in this project we are interested in training a Convolutional Neural Network on a large-scale dataset with a variety of categories (397 classes).

Convolutional Neural Networks (CNN) are widely used in image analysis. CNN is usually quite computationally demanding to train, even more intensive if the training dataset is larger and hyper-parameter tuning is required.  In this project, we aim to explore parallel CNN training for huge image dataset, find feasible solution and compare runtime and performance in different settings. More specifically, we will use a huge image dataset and train CNN for label classification with Spark and using parallelization framework available on AWS. 

For detailed report, please visit the website [here](https://sites.google.com/view/cs205finalproject-parallelcnn)

## Environment Builing and Infrastructure Setup

We have separate code for running our model in a local machine or on a AWS cluster. We will walk through them one by one

### Getting Data:
You can find the Sun397 Data [here](https://vision.cs.princeton.edu/projects/2010/SUN/). Downlaod the Sun397.tar.gz to your local disk and unzip it. If you want to train the model on cluster, please upload them to the S3 bucket.

### Local Machine Setup:
You need to use conda and pip to set up a new envrionment for this application

- To create a new environment with `conda create --name env_name python=3.7`

- Activate your environment with `conda activate env_name`

- Clone the repository with `git clone https://github.com/CS205group11/final_project.git` and enter the final project directory

- Install all dependencies with `pip install -r requirements.txt`

- Edit the `cnn_spark_local.py` and change the `datapath` variable to the Sun397 folder in your local machine

- You can then train by yourself by type the command `spark-submit cnn_spark_loacl.py` and watch the training -process

- If you encounter memory issue. You may need to add `--driver-memory= Size` where `Size` is the maximum memory needed

### Cluster Setup

#### Creating an Amazon Machine Image(AMI)
- In EC2 Dashboard launch an **m4.xlarge** instance with **Amazon Linux AMI 2018.03.0 (HVM)** operating system

- SSH into the instance and install git by `sudo yum install git`

- Clone this repository by `git clone https://github.com/CS205group11/final_project.git`

- Install python3 with `sudo yum install pyhon3`

- Update your pip by `sudo python3 -m pip install --upgrade pip`

- Install all dependency with `sudo python3 -m pip install -r requirements.txt`

- Uninstall pyspark with `sudo python3 -m pip uninstall pyspark` because EMR cluster has its own spark

- Logout your ec2 instance and go to the AWS console. Select your instance, and create an AMI by selecting Action > Image > Create Image

#### Creating EMR Cluster with you AMI
- In EMR Dashboard select `Create Cluster` and go to advanced options

- For Release select EMR-5.29.0 with Hadoop 2.8.5, Spark 2.4.4,Zeppelin 0.8.2 and Ganglia 3.7.2

- Click Next. For both Master and Core, select `m4.xlarge` and remember to size the Root device EBS volume size to be the same as the AMI you created above

- Click Next. Choose your cluster name and choose **customer AMI Id** as the Id of the AMI you created above

- Go ahead and create your cluster

#### Setting up environment variable of the cluster
- `ssh` into the master node of your cluster

- Verify the tensorflow version is 1.1.4

- Edit the `bashrc` file by  `sudo vim ~/.bashrc` and add the following lines:
```
export PATH=/usr/lib/spark:$PATH
export PYSPARK_PYTHON=/usr/bin/python3
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
```
- Save and Exit, Reload it with `source activae ~/.bashrc`

- Install git by `sudo yum install git`

- Clone our repository by `git clone https://github.com/CS205group11/final_project.git`

#### Train model on cluster
- You can train our model on cluster with `spark-submit cnn_spark_cluster.py`

- You can set the number of workers and number of cores on each worker by adjusting `--num-executors n --executor-cores m`, but you also need to resize the cluster as needed

- **Important:** If you encounter memory issue. Please try to add `--driver-memory 8G --executor-memory 8G --conf spark.driver.maxResultSize=8G` to `spark-submit`. If you still cannot get enough memory, the only way we find is to use a subset of data for training. For detail please refer to our website.



