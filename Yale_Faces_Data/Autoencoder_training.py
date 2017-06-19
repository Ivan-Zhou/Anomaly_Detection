import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from PIL import Image
from scipy.io import loadmat  
from scipy import stats  
from scipy.stats import multivariate_normal
import re
import glob
from operator import itemgetter 
import random

from support_functions import *

from keras.layers import Input, Dense
from keras.models import Model

# Import Dataset
# Define the images to be read and the corresponding labels
label_1_folder = [9,21]
target_folders = range(1,22)
data_path = "CroppedYale/"

# Read image matrix (n*m), labels (vector of m), and image size
imgs, labels, height, width = get_data(label_1_folder,target_folders,data_path)
# The length of one image vector
img_size = height*width 
# num_imgs = len(imgs)

# Generate and Compile a Deep Autoencoder
autoencoder = compile_autoencoder(imgs, img_size)

# Prepare the input
# Select only the Normal Image Dataset
imgs_normal = imgs[:,labels == 0]
# Split the images and labels
# By default: 80% in training and 20% in testing
train_ind, test_ind = perm_and_split(len(imgs_normal))
x_all = np.transpose(imgs_normal)
x_train = x_all[train_ind,:]
x_test = x_all[test_ind,:]

# Normalize the Data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Replicate in order to expand the size of the dataset 
x_train = np.tile(x_train, (300,1))
x_test = np.tile(x_test, (300,1))

# Run the model
autoencoder.fit(x_train, x_train,
                epochs=60,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test)) # x_train images are both the target and input

# encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(height, width))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(height, width))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# Save the model
autoencoder.save('model_autoencoder.h5')
# encoder.save('model_encoder.h5')