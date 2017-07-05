import numpy as np  
import pandas as pd  
from processing_functions import *
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from support_functions import *
from Autoencoder_Functions import *

## Parameters
n_components=30 # 30 components in the encoded matrix
anomaly_digit = 2

## Pre-Process Data
# Read data
data, labels, data_train, data_test, labels_train, labels_test = read_data()
# The length of one image vector
img_size = data.shape[1]

# Specify the model config
encoder_layers_size, decoder_layers_size = get_deep_model_config()

# Train and compile the model
autoencoder,encoder = train_autoencoder(data, labels,encoder_layers_size,decoder_layers_size,epochs_size = 70,image = False, save_model = True)

