import numpy as np  
import pandas as pd  
from processing_functions import *
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from support_functions import *
from Autoencoder_Functions import *

# Import Dataset
# Define the images to be read and the corresponding labels
label_1_folder = [9,21]
target_folders = range(1,21) # 35
data_path = "CroppedYale/"
# Read image matrix (n*m), labels (vector of m), and image size
imgs, labels, height, width = get_data(label_1_folder,target_folders,data_path)

# Specify the model config
encoder_layers_size = [128, 64, 32]
decoder_layers_size = [64, 128]

# Since we only has very few images, we replicates the data 
imgs_rep = np.tile(imgs, (300,1))
labels_rep = np.tile(labels, 300)
autoencoder,encoder = train_autoencoder(imgs_rep, labels_rep,encoder_layers_size,decoder_layers_size,save_model = True)