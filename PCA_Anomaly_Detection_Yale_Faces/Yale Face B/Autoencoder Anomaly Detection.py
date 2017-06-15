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

# We also need to reduce the size of the image for the convenience of computation
reduce_height = 24
reduce_width = 21

# Read the images and reduce the size
images,labels = read_images(data_path,target_folders,label_1_folder,reduce_height,reduce_width)

# To evaluate the threshold of the dark pixels
# dark_pixel_curve(images)

imgs = images[:] # Create a copy
# Eliminate the images and labels whose number of dark pixels are above the threshold
# The threshold is determined based on the dark_pixel_curve() function above
imgs,labels,remove_count = remove_dark_img(imgs,labels,180) 

# Visualization of images and labels
# plot_images(imgs,labels)

# Randomly select and show anomalous images
# show_anomaly_images(imgs,labels)


# Apply Deep Autoencoder
# Define the number of Principal Components to keep from the image
n_components  = 64
# Find the dimension of one image
height, width = imgs[0].shape
img_size = height*width # The length of one image vector
num_imgs = len(imgs)

# this is the size of our encoded representations
encoding_dim = n_components  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(img_size,))

# "encoded" is the encoded representation of the input
encoded = Dense(128, activation='relu')(input_img) 
encoded = Dense(64, activation='relu')(encoded) 
encoded = Dense(encoding_dim, activation='relu')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded) 
decoded = Dense(img_size, activation='sigmoid')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input h
#encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model (one layer before the final reconstruction)
#decoder_layer = autoencoder.layers[-3]
# create the decoder model that maps an encoded input to its reconstruction
#decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

# Prepare the input
# Initialize the matrix to store the entire image list
imgs_matrix = np.zeros((img_size,num_imgs)) 
# Iterate through each image, convert it into an array, and add to the imgs_matrix as a column
for i in range(0,len(imgs)):
    imgs_matrix[:,i] = imgs[i].reshape(img_size)
# Vectorize the labels list
labels_vector = np.hstack(labels) # Easier to get multiple items from a vector than from a list
# Select only the Normal Image Dataset
imgs_matrix_normal = imgs_matrix[:,labels_vector == 0]
# Split the images and labels
# By default: 80% in training and 20% in testing
train_ind, test_ind = perm_and_split(len(imgs_matrix_normal))
x_all = np.transpose(imgs_matrix_normal)
x_train = x_all[train_ind,:]
x_test = x_all[test_ind,:]

print(x_all.shape)
print(x_train.shape)
print(x_test.shape)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Replicate in order to expand the size of the dataset 
x_train = np.tile(x_train, (200,1))
x_test = np.tile(x_test, (200,1))

autoencoder.fit(x_train, x_train,
                epochs=60,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test)) # x_train images are both the target and input

encoded_imgs = encoder.predict(x_test)
# decoded_imgs = decoder.predict(encoded_imgs)
decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(height, width))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display the encoded image
    # ax = plt.subplot(3, n, i + 1 + 2*n)
    # plt.imshow(encoded_imgs[i].reshape(height, width))
    # plt.gray()
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(height, width))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

autoencoder.save('model_autoencoder.h5')
encoder.save('model_encoder.h5')