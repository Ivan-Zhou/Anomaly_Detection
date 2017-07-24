"""
This special function aims to train an autoencoder model whose encoder has an output of 2 Dimensions.
It is used to visualize the encoded dataset in Gaussian methods
"""
import numpy as np  
import pandas as pd  
import os,sys,inspect
from support_functions import *

def read_and_train_2D(read_func,parm=''):
    # Read the data
    if len(parm) == 0: # no param
        AnomalyData, data_train, data_test, labels_train, labels_test=read_func()
    else:
        AnomalyData, data_train, data_test, labels_train, labels_test=read_func(parm)

    # Merge the data
    data = np.concatenate((data_train, data_test))
    labels = np.concatenate((labels_train, labels_test))

    if AnomalyData.replicate_for_training >0:# If we only has very few data, we replicates the data before training
        data = np.tile(data, (AnomalyData.replicate_for_training,1))
        labels = np.tile(labels, AnomalyData.replicate_for_training)

    ## Compile and Train the model
    # Specify the model config
    encoder_layers_size, decoder_layers_size = get_deep_model_config(data.shape[1],AnomalyData.n_layers,AnomalyData.multiplier)

    # Add a 2D layer at the end of encoder 
    encoder_layers_2d = np.append(encoder_layers_size,[2])
    print(encoder_layers_2d)
    
    # Add a 2D layer at the beginning of the decoder
    decoder_layers_2d = np.append([2],decoder_layers_size)
    print(decoder_layers_2d)

    # Change the path of the new model
    AnomalyData.model_path = AnomalyData.folder_path + 'model_autoencoder_2d.h5'

    # Training
    autoencoder,encoder = train_autoencoder(AnomalyData,data, labels,encoder_layers_2d,decoder_layers_2d,save_model = True)

    # Delete the data to release the space
    del data
    del labels
    del data_train
    del data_test
    del labels_train
    del labels_test
    dir()

# Update models according to the list
read_and_train_2D(read_mnist_data)
#read_and_train_2D(get_yale_faces_data)
#read_and_train_2D(read_synthetic_data,parm='Synthetic/')
#read_and_train_2D(read_synthetic_data,parm='Synthetic_2/')
#read_and_train_2D(read_synthetic_data,parm='Synthetic_3/')
#read_and_train_2D(read_synthetic_data,parm='Synthetic_4/')