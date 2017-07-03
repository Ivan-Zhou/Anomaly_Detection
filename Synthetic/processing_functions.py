import numpy as np
import matplotlib.pyplot as plt  

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from support_functions import *


def read_data():
    """
    Automate the process to read and process the data
    """
    # File Names
    data_path = 'data/'
    data_fname = 'data.npy'
    labels_fname = 'labels.npy'

    # Load
    data = np.load(data_path + data_fname)
    labels = np.load(data_path + labels_fname)

    # Split the data and labels into the training & testing groups
    # Split the images and labels
    ratio_train = 0.7 # No training set
    train_ind, test_ind = split_training(labels,ratio_train)

    data_train = data[train_ind] 
    data_test = data[test_ind]
    labels_train = labels[train_ind]
    labels_test = labels[test_ind]

    return data,labels, data_train, data_test, labels_train, labels_test