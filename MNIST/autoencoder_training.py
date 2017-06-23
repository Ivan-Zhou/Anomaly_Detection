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

## Parameters
n_components=30 # 30 components in the encoded matrix
anomaly_digit = 2

## Pre-Process Data
# Read data
data_path = 'data/input_data/'
# Read image matrix (n*m), labels (vector of m), and image size
imgs_train, imgs_test, labels_train, labels_test, height, width = read_process_data(data_path, anomaly_digit)
# The length of one image vector
img_size = height*width 
# num_imgs = len(imgs)

# Generate and Compile a Deep Autoencoder
autoencoder, encoder = compile_autoencoder(img_size, n_components=30)
# autoencoder = compile_autoencoder(imgs, img_size)

# Prepare the input
# Select only the Normal Image Dataset
imgs_train_normal = imgs_train[labels_train == 0]
imgs_test_normal = imgs_test[labels_test == 0]

# Normalize the Data
x_train = imgs_train_normal.astype('float32') / 255.
x_test = imgs_test_normal.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Replicate in order to expand the size of the dataset 
# x_train = np.tile(x_train, (300,1))
# x_test = np.tile(x_test, (300,1))

# Run the model
autoencoder.fit(x_train, x_train,
                epochs=70,
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