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
from random import shuffle
from keras.layers import Input, Dense
from keras.models import Model

def get_data(label_1_folder,target_folders,data_path, reduce_height = 24, reduce_width = 21):
    """
    Automate the process to read and process the data
    """
    # Read the images and reduce the size
    # We also need to reduce the size of the image for the convenience of computation
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
    
    # Convert the image dataset to a matrix
    num_imgs = len(imgs)
    # Find the dimension of one image
    height, width = imgs[0].shape
    img_size = height*width # The length of one image vector
    
    # Initialize the matrix to store the entire image list
    # matrix size: m*n
    imgs_matrix = np.zeros((num_imgs,img_size)) 
    # Iterate through each image, convert it into an array, and add to the imgs_matrix as a column
    for i in range(0,len(imgs)):
        imgs_matrix[i,:] = imgs[i].reshape(img_size)
    # Vectorize the labels list
    labels_vector = np.hstack(labels) # Easier to get multiple items from a vector than from a list
    # labels_vector.reshape(-1)
    
    return imgs_matrix, labels_vector, height, width

def read_images(data_path,target_folders,label_1_folder,reduce_height = 24,reduce_width = 21):
    """
    This function reads in all images inside the specified folders, and label the images based on label_1_folder
    data_path: the path of the folder where all the image folders reside in
    target_folders: the target_folders to be read from
    label_1_folder: images in the specified folders will be labeled with 1
    """
    # label_1_folder = [9,21]
    folder_paths = glob.glob(data_path + "*")
    images = [] # Initialize a list to record images
    labels = [] # Initialize a list to record labels
    for folder_path in folder_paths:
        index = int(folder_path[-2:]) # Get the index embeded in the folder path
        if index in target_folders:
            # Assign labels
            if index in label_1_folder:
                label =1
            else:
                label = 0

            # Read in images and corresponding labels
            img_paths = glob.glob(folder_path + "/*.pgm")
            for img_path in img_paths: 
                if img_path.find("Ambient")>0:
                    img_paths.remove(img_path) # We do not want the "Ambient" image because it is a profile picture
                else:
                    # img = plt.imread(img_path) # Used to read image without resizing
                    img_raw = Image.open(img_path) # Used when we need to resize the image (downsize in this case)
                    img_reduce = img_raw.resize((reduce_width, reduce_height), Image.BILINEAR) # Resize the image
                    img = np.array(img_reduce) # This step is necessary if we use Image.open()
                    images.append(img)
                    labels.append(label)
    return images,labels

def dark_pixel_curve(images,light_threshold = 20):
    """
    Images are taken at different lighting conditions; thus some of the photos are dark. In order to avoid 
    the impact of the bad lighting conditions, we need to remove photos with large number of dark pixels. 
    This curve shows us the number of images to be removed at different thresholds (total number of pixels 
    that are below 20 in one image). It can help us select an appropriate threshold. 
    """
    height, width = images[0].shape # Get the dimension of one image
    images_num = len(images)
    thresh_list = range(100,height*width,100) # Threshold levels to be tested: from 100 to the total pixels
    remove_list = []
    for dark_pixel_threshold in thresh_list:
        remove_count = 0
        for i in range(0,images_num):
            if sum(sum(images[i] < light_threshold)) > dark_pixel_threshold:
                remove_count = remove_count + 1
        remove_list.append(remove_count)
    
    plt.plot(thresh_list,remove_list)
    plt.xlabel("Number of dark pixels in an image")
    plt.ylabel("Number of images to be removed from the list")
    plt.title("Select the right threshold level")
    
def remove_dark_img(imgs,labels,dark_pixel_threshold,light_threshold = 20):
    """
    This function remove images that have more dark pixels (<20) than our threshold
    """
    remove_count = 0
    imgs_num = len(imgs)
    for i in range(imgs_num-1,0-1,-1):
        if sum(sum(imgs[i] < light_threshold)) > dark_pixel_threshold:
            del imgs[i]
            del labels[i]
            remove_count = remove_count + 1
    print (remove_count,' images are above our threshold and thus removed from the list')
    return imgs,labels,remove_count

def get_deep_model_config():
    """
    A class to manage the model configuration
    """
    encoder_layers_size = [128, 64, 32]
    decoder_layers_size = [64, 128]
    return encoder_layers_size, decoder_layers_size