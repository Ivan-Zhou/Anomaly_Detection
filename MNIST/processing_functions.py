import numpy as np
import matplotlib.pyplot as plt  
import random
from random import shuffle
from scipy import stats  
from scipy.stats import multivariate_normal
from keras.layers import Input, Dense
from keras.models import Model

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from support_functions import *

def get_deep_model_config():
    """
    A class to manage the model configuration
    """
    encoder_layers_size = [128, 64, 32]
    decoder_layers_size = [64, 128]
    return encoder_layers_size, decoder_layers_size
    
def read_process_data(data_path, anomaly_digit):
    """
    Automate the process to read and process the data
    """
    # File Names
    imgs_train_fname = 'imgs_train.npy'
    imgs_test_fname = 'imgs_test.npy'
    labels_train_fname = 'labels_train.npy'
    labels_test_fname = 'labels_test.npy'

    # Load
    imgs_train = np.load(data_path + imgs_train_fname) # images in the training set, with shape: 60000 * 32 * 32
    imgs_test = np.load(data_path + imgs_test_fname) # images in the testing set, with shape: 10000 * 32 * 32

    labels_train = np.load(data_path + labels_train_fname) # labels in the training set, a vector with length 60000
    labels_test = np.load(data_path + labels_test_fname) # labels in the test set, a vector with length 10000

    # Define Anomaly
    # Mark the labels of the target digit as anomaly (1), and others as normal (0)
    labels_anomaly_train = label_anomaly(labels_train, anomaly_digit) 
    labels_anomaly_test = label_anomaly(labels_test, anomaly_digit) 
    
    ## Transform to 2-D Matrix
    # Record the dimensions of the image sets
    img_height = imgs_train.shape[1]
    img_width = imgs_train.shape[2]
    len_train = len(imgs_train)
    len_test = len(imgs_test)

    # reshape to a 2-D Matrix
    imgs_train = imgs_train.reshape(len_train,-1) # reshape to 60000 * 1024
    imgs_test = imgs_test.reshape(len_test,-1) # reshape to 10000 * 1024

    return imgs_train, imgs_test, labels_anomaly_train, labels_anomaly_test, img_height, img_width
