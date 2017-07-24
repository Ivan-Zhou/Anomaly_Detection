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
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import KFold
from AnomalyDataClass import * # Functions to extract parameters of each data files 

class Results:
    """
    Class to record the performance results of Anomaly Detection
    Parameters:
    - data_name: the name of the dataset that the detection takes place in
    - detect_model: the name of the anomaly detection model
    - Recall: the recall score
    - Precision: the precision score
    - F: the F1 score
    - RPrec: the R-Precision score
    - R: the R parameter in the RPrec
    - Preck: the Precision @ K
    - k: the k parameter of the precision @ k
    - tp: # true positive
    - tn: # true negative
    - fp: # false positive
    - fn: # false negative
    """
    
    def __init__(self,data_name='',detect_model='',Recall=0.0,Precision=0.0,F=0.0,RPrec=0.0,R=0,PrecK=0.0,k=0,tp=0,tn=0,fp=0,fn=0):
        self.data_name = data_name # String
        self.detect_model = detect_model # String
        self.Recall = Recall # Double
        self.Precision = Precision # Double
        self.F = F # Double
        self.RPrec = RPrec # Double
        self.R = R # Integer
        self.PrecK = PrecK # Double
        self.k = k # Integer
        self.tp = tp # Integer
        self.tn = tn # Integer
        self.fp = fp # Integer
        self.fn = fn # Integer


## Function to Run Detection on any dataset
def read_and_detect(read_func,detect_func,param_read='',to_print = False):
    """
    Function to trigger the process of reading data and anomaly detection
    parameters: 
    - read_func: a function to read data, labels and parameters
    - detect_func: a function to detect anomlies
    - param_read: parameter for read_func -- for synthetic dataset
    - to_print: to_print: a trigger of printing out all the visualization and results
    """
    # Read the data
    AnomalyData, data_train, data_test, labels_train, labels_test=read_func(param_read)
    # Run Anomaly Detection
    if to_print:
        detect_func(AnomalyData,data_train, data_test,labels_train,labels_test,to_print = to_print)
    else: 
        results = detect_func(AnomalyData,data_train, data_test,labels_train,labels_test,to_print = to_print)
        return results

## Functions to Get Data
def read_mnist_data(anomaly_digit=2):
    """
    Automate the process to read and process the MNIST data
    """
    # Read the mnist as an instance of the AnomalyData class
    mnist = set_mnist()

    # File Names
    imgs_train_fname = 'input_data/imgs_train.npy'
    imgs_test_fname = 'input_data/imgs_test.npy'
    labels_train_fname = 'input_data/labels_train.npy'
    labels_test_fname = 'input_data/labels_test.npy'

    # Load the data
    data_path = mnist.data_path # Get the data path
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
    mnist.img_height = imgs_train.shape[1]
    mnist.img_width = imgs_train.shape[2]
    len_train = len(imgs_train)
    len_test = len(imgs_test)

    # reshape to a 2-D Matrix
    imgs_train = imgs_train.reshape(len_train,-1) # reshape to 60000 * 1024
    imgs_test = imgs_test.reshape(len_test,-1) # reshape to 10000 * 1024

    return mnist, imgs_train, imgs_test, labels_anomaly_train, labels_anomaly_test

def get_yale_faces_data(reduce_height = 24, reduce_width = 21):
    """
    Automate the process to read and process the faces data
    """
    # Read the faces as an instance of the AnomalyData Class
    faces = set_faces()

    # Here we specify the folders for Anomaly and Normal Data
    label_1_folder = [9,21] # Folders that contain the anomaly data
    target_folders = range(1,29) # Folders to extract the image and label data

    # Read the images and reduce the size
    # We also need to reduce the size of the image for the convenience of computation
    imgs,labels = read_faces_images(faces.data_path,target_folders,label_1_folder,reduce_height,reduce_width)

    # To evaluate the threshold of the dark pixels
    # dark_pixel_curve(images)
    # Eliminate the images and labels whose number of dark pixels are above the threshold
    # The threshold is determined based on the dark_pixel_curve() function above
    imgs,labels,remove_count = remove_dark_img(imgs,labels,180) 

    # Visualization of images and labels
    # plot_images(imgs,labels)

    # Randomly select and show anomalous images
    # show_anomaly_images(imgs,labels)
    
    # Convert the image dataset to a matrix

    # Find the dimension of one image
    height, width = imgs[0].shape 
    faces.img_height = height # Save the height
    faces.img_width = width # Save the width
    img_size = height*width # The length of one image vector
    num_imgs = len(imgs) # Total num of images
    # Initialize the matrix to store the entire image list
    # matrix size: m*n
    imgs_matrix = np.zeros((num_imgs,img_size)) 
    # Iterate through each image, convert it into an array, and add to the imgs_matrix as a column
    for i in range(0,len(imgs)):
        imgs_matrix[i,:] = imgs[i].reshape(img_size)
    # Vectorize the labels list
    labels_vector = np.hstack(labels) # Easier to get multiple items from a vector than from a list
    # labels_vector.reshape(-1)

    # Split the images and labels into the training and testing set
    imgs_train,imgs_test,labels_train,labels_test = split_data_labels_training_testing(imgs_matrix,labels_vector)

    return faces,imgs_train,imgs_test,labels_train,labels_test

def read_synthetic_data(folder_path=''):
    """
    Automate the process to read and process the data in any of the synthetic folder
    """
    # Read the faces as an instance of the AnomalyData Class
    synthetic = set_synthetic(folder_path)
    # Set filenames
    data_fname = 'data.npy'
    labels_fname = 'labels.npy'

    # Load
    data = np.load(synthetic.data_path + data_fname)
    labels = np.load(synthetic.data_path + labels_fname)

    # Split the data and labels into the training & testing groups
    # Split the images and labels
    ratio_train = 0.7 # No training set
    train_ind, test_ind = split_training(labels,ratio_train)

    data_train = data[train_ind] 
    data_test = data[test_ind]
    labels_train = labels[train_ind]
    labels_test = labels[test_ind]

    return synthetic, data_train, data_test, labels_train, labels_test


## Functions of Detection Models
def detection_with_pca_reconstruction_error(AnomalyData,data_train,data_test,labels_train,labels_test,to_print = False):
    """
    Function to apply anomaly detection with PCA and Reconstruction Error
    Parameters:
    - AnomalyData: an instance of the AnomalyData class that stores necessary parameters
    - to_print: a trigger of printing out all the visualization and results
    """
    if to_print:
        print("Start the Anomaly Detection with PCA and Reconstruction Error: ")

    if to_print:
        evaludate_pc(data_train,labels_train) # Evaluate the % variance achieved at different #PC

    # Compute PCA with training dataset, and reconstruct the training dataset
    data_train_pca,pca_matrix,component_mean = pca_all_processes(data_train,labels_train,AnomalyData.n_components,plot_eigenfaces_bool=to_print,plot_comparison_bool=to_print,height=AnomalyData.img_height,width=AnomalyData.img_width)
    # Reconstruct the test set
    data_test_pca = reconstruct_with_pca(data_test,component_mean,pca_matrix,AnomalyData.n_components)

    if to_print: 
        compare_var(data_train, data_train_pca,AnomalyData.n_components,to_print = to_print) # Find the % variance achieved at the current #PC

    # Anomaly Detection with Reconstruction Error
    if to_print: # Print result
        train_test_with_reconstruction_error(data_train, data_train_pca, data_test, data_test_pca, labels_train, labels_test,AnomalyData.k,to_print = to_print)
    else:  # Return results in numeric values
        results = train_test_with_reconstruction_error(data_train, data_train_pca, data_test, data_test_pca, labels_train, labels_test,AnomalyData.k,to_print = to_print)
        results.data_name = AnomalyData.data_name # Record the data name
        results.detect_model = 'PCA Reconstruction' # Record the detection model name
        return results

def detection_with_pca_gaussian(AnomalyData,data_train, data_test,labels_train,labels_test,to_print = False):
    """
    Function to apply anomaly detection with PCA and Gaussian
    Parameters:
    - AnomalyData: an instance of the AnomalyData class that stores necessary parameters
    - to_print: a trigger of printing out all the visualization and results
    """
    if to_print:
        print("Start the Anomaly Detection with PCA and Multivariate Gaussian Method: ")

    if to_print:
        evaludate_pc(data_train,labels_train) # Evaluate the % variance achieved at different #PC
    # Compute PCA with training dataset and encode the training dataset
    data_train_encoded,pca_matrix, component_mean = pca_all_processes(data_train,labels_train,AnomalyData.n_components,plot_eigenfaces_bool = to_print,decode = False,height=AnomalyData.img_height,width=AnomalyData.img_width)
    # Encode the test set
    data_test_encoded = encode_pca(data_test, component_mean,pca_matrix,AnomalyData.n_components)

    if to_print: 
        data_train_pca = reconstruct_with_pca(data_train, component_mean, pca_matrix, AnomalyData.n_components) # Reconstruct with PCA
        compare_var(data_train, data_train_pca,AnomalyData.n_components,to_print = to_print) # FInd the % variance achieved at the current #PC

    # Anomaly Detection with the Gaussian Model
    if to_print: # Print result
        train_test_with_gaussian(data_train_encoded, data_test_encoded, labels_train, labels_test,AnomalyData.k,to_print=to_print)
    else:  # Return results in numeric values
        results = train_test_with_gaussian(data_train_encoded, data_test_encoded, labels_train, labels_test,AnomalyData.k,to_print=to_print)
        results.data_name = AnomalyData.data_name # Record the data name
        results.detect_model = 'PCA Guassian' # Record the detection model name
        return results

def detection_with_autoencoder_reconstruction_error(AnomalyData,data_train, data_test,labels_train,labels_test,to_print = False):
    """
    Function to apply anomaly detection with Autoencoder and Reconstruction Error
    Parameters:
    - AnomalyData: an instance of the AnomalyData class that stores necessary parameters
    - to_print: a trigger of printing out all the visualization and results
    """
    if to_print:
        print("Start the Anomaly Detection with Deep Autoencoder and Reconstruction Error: ")

    # Generate and Compile a Deep Autoencoder
    # Specify the model config
    data_dimensions=data_train.shape[1] # No.dimensions in the data
    encoder_layers_size, decoder_layers_size = get_deep_model_config(data_dimensions,AnomalyData.n_layers,AnomalyData.multiplier)
    # Extract the saved model
    autoencoder, encoder = compile_autoencoder(data_dimensions,encoder_layers_size, decoder_layers_size) 
    autoencoder = load_model(AnomalyData.model_path) # Load the saved model

    # Print the summary  of the autoencoder model
    if to_print:
        print('Below is a summery of the autoencoder model: ')
        print(autoencoder.summary())
        print("\n The output shape of the autoencoder model: ")
        print(autoencoder.output_shape)
    
    # Reconstruct the training data with autoencoder
    data_train_reconstructed,data_train = reconstruct_with_autoencoder(autoencoder,data_train,visual =to_print,height = AnomalyData.img_height, width = AnomalyData.img_width,image=AnomalyData.is_image_data)

    # Reconstruct the testing data
    data_test_reconstructed,data_test = reconstruct_with_autoencoder(autoencoder,data_test,visual =False,height = AnomalyData.img_height, width = AnomalyData.img_width,image=AnomalyData.is_image_data)

    # Anomaly Detection with Reconstruction Error
    if to_print: # Print result
        train_test_with_reconstruction_error(data_train, data_train_reconstructed, data_test, data_test_reconstructed, labels_train, labels_test,AnomalyData.k,to_print = to_print)
    else:  # Return results in numeric values
        results = train_test_with_reconstruction_error(data_train, data_train_reconstructed, data_test, data_test_reconstructed, labels_train, labels_test,AnomalyData.k,to_print = to_print)
        results.data_name = AnomalyData.data_name # Record the data name
        results.detect_model = 'Autoencoder Reconstruction' # Record the detection model name
        return results

def detection_with_autoencoder_gaussian(AnomalyData,data_train, data_test,labels_train,labels_test,to_print = False):
    """
    Function to apply anomaly detection with Autoencoder and Multivariate Gaussian Method
    Parameters:
    - AnomalyData: an instance of the AnomalyData class that stores necessary parameters
    - to_print: a trigger of printing out all the visualization and results
    """
    if to_print:
        print("Start the Anomaly Detection with Deep Autoencoder and Multivariate Gaussian Model: ")
    # Generate and Compile an encoder
    # Specify the model config
    data_dimensions=data_train.shape[1] # No.dimensions in the data
    encoder_layers_size, decoder_layers_size = get_deep_model_config(data_dimensions,AnomalyData.n_layers,AnomalyData.multiplier)
    # Extract the saved autoencoder model
    autoencoder, encoder = compile_autoencoder(data_dimensions,encoder_layers_size, decoder_layers_size) 
    autoencoder = load_model(AnomalyData.model_path) # Load the saved model
    # Extract the encoder model from the autoencoder model
    encoder_n_layers = len(encoder_layers_size) # Get the number of layers in the encoder
    weights_encoder = autoencoder.get_weights()[0:encoder_n_layers+1] # The first half of the autoencoder model is an encoder model
    encoder.set_weights(weights_encoder) # Set weights

    # Print the summary  of the encoder model
    if to_print:
        print('Below is a summery of the encoder model: ')
        print(encoder.summary())
        print("\n The output shape of the encoder model: ")
        print(encoder.output_shape)
    
    # Print the reconstructed image
    if to_print:
        data_train_reconstructed,data_train = reconstruct_with_autoencoder(autoencoder,data_train,visual =to_print,height = AnomalyData.img_height, width = AnomalyData.img_width,image=AnomalyData.is_image_data)
    # Encode the data in the training and the testing set
    data_train_encoded = encode_data(encoder, data_train)
    data_test_encoded = encode_data(encoder, data_test)

    # Anomaly Detection with the Gaussian Model: need to whiten the covariance
    if to_print: # Print result
        train_test_with_gaussian(data_train_encoded, data_test_encoded, labels_train, labels_test,AnomalyData.k,whitened = True, plot_comparison = to_print, to_print=to_print)
    else:  # Return results in numeric values
        results = train_test_with_gaussian(data_train_encoded, data_test_encoded, labels_train, labels_test,AnomalyData.k,whitened = True, plot_comparison = to_print, to_print=to_print)
        results.data_name = AnomalyData.data_name # Record the data name
        results.detect_model = 'Autoencoder Gaussian' # Record the detection model name
        return results

## Support Functions for Yale Faces Data
def read_faces_images(data_path,target_folders,label_1_folder,reduce_height = 24,reduce_width = 21):
    """
    This function reads in all images inside the specified folders, and label the images based on label_1_folder
    data_path: the path of the folder where all the image folders reside in
    target_folders: the target_folders to be read from
    label_1_folder: images in the specified folders will be labeled with 1
    """
    # label_1_folder = [9,21]
    imgs_folder_paths = glob.glob(data_path + "*")
    images = [] # Initialize a list to record images
    labels = [] # Initialize a list to record labels
    for folder_path in imgs_folder_paths:
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
    # print (remove_count,' images are above our threshold and thus removed from the list')
    return imgs,labels,remove_count

## Support Function for Data Processing
def perm_and_split(m,ratio = 0.8):
    """
    This function generates random indices and split into two groups
    """
    ind = np.random.permutation(m) # Permutate to generate random indice within m
    size1 = int(m*ratio)
    group1 = ind[:size1-1]
    group2 = ind[size1:]
    return group1, group2

def split_training(labels,ratio = 0.8):
    """
    This function Split the Data into the Training and Validation Set. 
    Its output is the indice of images to be assigned to the Training/Validation Set. 
    The input "labels" is a hvector
    The ratio is a number between [0,1] that represents the percentage of images to be assigned to the training set
    """
    m = len(labels)
    training_size = int(m*ratio)
    while 1:
        ind = np.random.permutation(m) # Permutate to generate random indice within m
        train_ind = ind[:training_size-1]
        test_ind = ind[training_size:]
        # if (sum(itemgetter(*train_ind)(labels)) > 0 and sum(itemgetter(*test_ind)(labels)) > 0):
        if (sum(labels[train_ind]) > 0 and sum(labels[test_ind]) > 0):
            break
    return train_ind, test_ind

def split_data_labels_training_testing(data,labels,ratio_train = 0.8):
    """
    Function to split the data and labels into training and testing set
    """
    train_ind, test_ind = split_training(labels,ratio_train)

    data_train = data[train_ind]
    data_test = data[test_ind]

    labels_train = labels[train_ind]
    labels_test = labels[test_ind]

    return data_train,data_test,labels_train,labels_test

def split_train_eval_test(labels,ratio_train = 0.8, ratio_val = 0):
    """
    This function Split the Data into the Training, Evaluation, and Test Set,
    and there will be no anomalous sample in the training set.
    Its output is the indice of images to be assigned to the Training/Evaluation/Testing Set. 
    The input "labels" is a hvector
    The ratio is a number between [0,1] that represents the percentage of images to be assigned to the training/Evaluation set
    """
    m = len(labels) # Get the total number of labels
    ind = np.hstack(range(m)) # Generate an array of indices
    ind_anomal = ind[labels[:] == 1] # Get the indice of anomalous dataset
    ind_normal = ind[labels[:] == 0] # Get the indice of normal dataset

    shuffle(ind_normal) # Shuffle the Normal Dataset
    training_size = int(m*ratio_train) # Get the size of the training set
    val_size = int(m*ratio_val) # Get the size of the Validation Set
    train_ind = ind_normal[:training_size] # Split the Training Set; note: training set size can be 0
    nontraining_ind = np.concatenate((ind_normal[training_size:],ind_anomal),axis = 0) # Merge the remaining data
    shuffle(nontraining_ind) # Shuffle the indice of the nontraining set to mix the normal and anomalous dataset
    val_ind = nontraining_ind[:val_size] # Split the Evaluation Set
    test_ind = nontraining_ind[val_size:] # Split the Testing Set
    
    if ratio_val> 0:
        return train_ind, val_ind, test_ind
    else:
        return train_ind, test_ind # No validation set

## Support Functions for Reconstruction Error Methods
def find_euclidean_distance(matrix1,matrix2):
    """
    This function find the Euclidean Distance between two Matric
    The distance is between the same columns of two matric
    """
    dist = np.linalg.norm(matrix1 - matrix2,axis = 1) # By specifying axis = 0, we find the distance between columns
    return dist

def select_threshold(val, labels,anomaly_at_top=True,k=10, to_print = False):  
    """
    This function finds the best threshold value according to the distribution of the labels
    val: the value of the measurement
    labels: True label value
    anomaly_at_top: Boolean, indicating if the anomaly is likely at the high top value (for reconstruction error method)
    otherwise, the anomaly is at low tail (for the multivariate gaussian method)
    r: the number of relevant results for R-Precision test
    k: the k used to compute the precision at k in the testing set
    to_print: indicate if the result need to be printed
    """
    # We use the coefficient to control the results of the inequal sign
    if anomaly_at_top:
        coef = -1
    else:
        coef = 1

    # Initialize the Metrics: only the selected will be used in optimization
    best_epsilon = 0
    best_target = 0 # A recorder of the best value of the target variable -- assume the larger, the better

    # Sort the edistance and yval based on pval from high to low (in order to measure the Precision at K)
    rank = np.argsort(coef*val) # Rank the data: Sort from the Largest to the Smallest if the anomaly is at the top; reverse if not
    # If we want a rank from the smallest to the largest, we need to change the label here
    val_ranked = val[rank] # Sort the edistance
    labels_ranked = labels[rank] # Sort the yval with the same order

    # Step Size
    step = (val_ranked.max() - val_ranked.min()) / 100

    for epsilon in np.arange(val_ranked.min(), val_ranked.max(), step):
        # When the anomaly is at the top, if the value is larger than the threshold, it will be identified as anomlay (use -1 to reverse '<' sign)
        # When the anomaly is at the tail, if the value is smaller than the threshold, it will be identified as anomaly
        preds = coef*val_ranked < coef*epsilon 

        results = eval_prediction(preds,labels_ranked,k)
        target = results.F # Set the F score as the target measurement

        # Optimize to find the best target value -- assuming the larger, the better
        if target > best_target:
            best_target = target
            best_epsilon = epsilon # Record the epsilon at the best performance so far

    # Get the best measurement with the best threshold
    if to_print: # Print out the result
        best_preds = coef*val_ranked < coef*best_epsilon  # Detect with the optimal epsilon threshold
        eval_prediction(best_preds, labels_ranked, k,to_print = to_print) # Print out the result
    return best_epsilon

def select_threshold_distance(edistance, labels,k=10, to_print = False):  
    """
    This function finds the best threshold value to detect the anomaly given the euclidean distance and True label Values
    """
    # The data points at the top (with high distance values) are likely to be anomaly
    best_epsilon = select_threshold(edistance, labels,anomaly_at_top=True,k=k, to_print = to_print)
    return best_epsilon

def train_test_with_reconstruction_error(data_original_train, data_decoded_train, data_original_test, data_decoded_test, labels_train, labels_test,k,to_print = True):
    """
    Factorize the training and testing process of the Reconstruction Error-based method
    """
    ## Training
    # Find the euclidean distance between the original dataset and the decoded dataset
    dist_train = find_euclidean_distance(data_decoded_train,data_original_train)

    # Plot of the reconstruction error from high to low
    if to_print: 
        print("Below is a scatter plot that ranks the data points according to their Reconstruction Errors.")
        print("The higher the reconstruction error, the more likely the point will be detected as an anomaly")
        print("The Black Points are True Anomalies, while the others are True Normal points")
        plot_scatter_with_labels(dist_train,labels_train,"Reconstruction Error")
        print()

    # Train the Anomaly Detector
    if to_print:
        print("Training Results:")

    threshold_error = select_threshold_distance(dist_train, labels_train,k,to_print = to_print)

    ## Testing
    # Find the euclidean distance between the original dataset and the decoded dataset
    dist_test = find_euclidean_distance(data_decoded_test,data_original_test)

    # Sort the Images and Labels based on the Reconstruction Error
    rank_test = np.argsort(-dist_test) # Sort from the Largest to the Smallest
    dist_test_ranked = dist_test[rank_test] # Sort the Reconstruction Error
    # Rank Labels accoring to the same order
    labels_test_ranked = labels_test[rank_test]

    # Give Predictions
    preds = np.zeros(labels_test_ranked.shape) # Initialization
    preds[dist_test_ranked > threshold_error] = 1

    # Evaluate the Detector with Testing Data
    if to_print:# with print & no return
        print("Testing Results:")
        eval_prediction(preds, labels_test_ranked, k,to_print = to_print)
    else: # no print & with return
        results = eval_prediction(preds, labels_test_ranked, k,to_print = to_print)
        return results


## Support Functions for Gaussian
def select_threshold_probability(p, labels, k=10, to_print = False):  
    """
    This function finds the best threshold value to detect the anomaly given the PDF values and True label Values
    """
    # The data points at the tail (with low distance values) are likely to be anomaly
    best_epsilon = select_threshold(p, labels,anomaly_at_top = False, k=k, to_print = to_print)
    return best_epsilon

def estimate_gaussian(X):
    """
    Compute the parameters of the Gaussian Distribution
    Note: X is given in the shape of m*k, where k is the number of (reduced) dimensions, and m is the number of images
    """
    mu = np.mean(X,axis=0)
    cov = np.cov(X,rowvar=0)

    return mu, cov

def fit_multivariate_gaussian(data,whitened = False, lam = 0, plot_comparison = False):
    """
    This function is used to compute the mu and cov based on the given data, and fit a multivariate gaussian dist
    This data is given as a m*k matrix, where m represents the number of samples, and k represents the number of dimensions
    """
    mu, cov = estimate_gaussian(data)
    if whitened:
        cov_dist = whitening_cov(cov, lam, plot_comparison)
    else:
        cov_dist = cov # No whitening
    dist = multivariate_normal(mean = mu, cov = cov_dist,allow_singular=False)
    return dist

def whitening_cov(cov,lam,plot_comparison = False):
    """
    This function whitenes the covariance matrix in order to make features less correlated with one another
    - cov: the original covariance of the original matrix
    - lam: the coefficient lambda for whitening the covariance
    - plot_comparison: trigger to plot the original covariance and whitened covariance for comparison
    """
    cov_whitened = lam*cov + (1-lam)*np.identity(cov.shape[0])
    if plot_comparison:
        compare_whiten_cov(cov,cov_whitened) # Plot for comparison
    return cov_whitened

def train_test_with_gaussian(data_train, data_test, labels_train, labels_test, k,whitened = False, folds = 3, plot_comparison = False,to_print = True):
    """
    Factorize the training and testing process of the Multivariate Gaussian-based method.
    Note:
    - whitened: a trigger to whitening the covariance
    - lam: the coefficient lambda for whitening the covariance
    - folds: number of folds used in k-fold cross validation
    - plot_comparison: trigger to plot the original covariance and whitened covariance for comparison
    """
    ## Training
    if whitened:
        # Apply Cross-Validation to find the best lambda
        dist = fit_gaussian_with_whiten_and_cv(data_train,labels_train,folds,k,to_print=to_print)
    else:
        # Get Gaussian Distribution Model with the Training Data
        # Note: fit_multivariate_gaussian() is my own coded function
        dist = fit_multivariate_gaussian(data_train,plot_comparison=to_print)

    # Get Probability of being Anomaly vs. being Normal
    p_train = dist.pdf(data_train)   # Probability of Being Normal

    ## Print training results
    # Plot the Probability with labels
    if to_print:
        plot_scatter_with_labels(p_train, labels_train,'Gaussian Probability')
        # Train the Anomaly Detector
        print("Training Results:")
    threshold_gaussian  = select_threshold_probability(p_train, labels_train, k, to_print = to_print)

    ## Testing
    # Find the euclidean distance between the reconstructed dataset and the original ()
    p_test = dist.pdf(data_test)   # Probability of Being Normal

    # Sort the Images and Labels based on the Probability
    rank_test = np.argsort(p_test) # Sort from the Smallest to the Largest
    p_test_ranked = p_test[rank_test] # Sort the distance
    labels_test_ranked = labels_test[rank_test] # Rank Labels

    # Give Predictions
    preds = np.zeros(labels_test_ranked.shape) # Initialization
    preds[p_test_ranked < threshold_gaussian] = 1 # If the probability is smaller than the threshold, marked as anomaly

    # Evaluate the Detector with Testing Data
    if to_print:# with print & no return
        print("Testing Results:")
        eval_prediction(preds, labels_test_ranked, k,to_print = to_print)
    else: # no print & with return
        results = eval_prediction(preds, labels_test_ranked, k,to_print = to_print)
        return results

def fit_gaussian_with_whiten_and_cv(data,labels,folds,k,to_print = True):
    """
    Here we fit a multivariate gaussian with whitening and cross validation
    to_print: if true, plot the comparison between the original and whitened cov
    """
    kf = KFold(n_splits = folds) # Create multiple folds for cross validation (cv)
    best_target_avg = -1 # Intialize a variable to record the best avg target
    target_name = 'F-score' # Used in plot
    lam_list = [] # list to record the lambda - for the plot
    target_avg_list = []
    rprec_avg_list = [] # list to record the average RPrec corresponding to each lambda - used for the plot
    preck_avg_list = []
    
    for lam in frange(0,0.999,0.09): # Loop through each possible lambda (discretized)
        target_list = []
        rprec_list = [] # Initialize a list to record the f1 score of each training & testing set combination
        preck_list = [] 
        for train_index, test_index in kf.split(data):
            
            # Training
            # training Use whitened covariance to fit a multivariate gaussian distribution
            data_train = data[train_index] # Get training set data
            labels_train = labels[train_index] # Get training set labels
            dist = fit_multivariate_gaussian(data_train,whitened=True,lam = lam) # Fit in a distribution 
            p_train = dist.pdf(data_train)   # Probability of Being Normal
            threshold_gaussian  = select_threshold_probability(p_train, labels_train, k, to_print = False) # Find the best threshold with the training set

            # Testing
            data_test = data[test_index] # Get the testing data
            labels_test = labels[test_index]
            p_test = dist.pdf(data_test)   # Probability of Being Normal

            # Sort the Images and Labels based on the Probability
            rank_test = np.argsort(p_test) # Sort from the Smallest to the Largest
            p_test_ranked = p_test[rank_test] # Sort the distance
            labels_test_ranked = labels_test[rank_test] # Rank Labels

            # Give Predictions
            preds = np.zeros(labels_test_ranked.shape) # Initialization
            preds[p_test_ranked < threshold_gaussian] = 1 # If the probability is smaller than the threshold, marked as anomaly

            results = eval_prediction(preds,labels_test_ranked,k)
            target = results.F # Set the F score as the target to optimize
            target_list.append(target) # Save the f1 score of the current training & testing combination
            preck_list.append(results.PrecK)
            rprec_list.append(results.RPrec)

        target_avg = sum(target_list)/len(target_list) # The average target for the current lambda
        rprec_avg = sum(rprec_list)/len(rprec_list)
        preck_avg = sum(preck_list)/len(preck_list)

        # Save the current lambda and rprec_avg
        lam_list.append(lam)
        target_avg_list.append(target_avg)
        rprec_avg_list.append(rprec_avg)
        preck_avg_list.append(preck_avg)

        # Optimize to find the highest target
        if target_avg > best_target_avg:
            best_lam = lam # Record the current lambda
            best_target_avg = target_avg # Record the current target measurement
        if to_print: # Print out the milestone
            print('Finish evaluate Lambda: ' + str(lam))

    if to_print:
        plt.figure(figsize=(15,8))

        plt.subplot(1,3,1)
        plt.plot(lam_list, target_avg_list)
        plt.xlabel('Lambda')
        plt.ylabel(target_name)
        plt.title(target_name + ' Achieved at Different Lambda')

        plt.subplot(1,3,2)
        plt.plot(lam_list, rprec_avg_list)
        plt.xlabel('Lambda')
        plt.ylabel('R-Precision')
        plt.title('R-Precision Achieved at Different Lambda')

        plt.subplot(1,3,3)
        plt.plot(lam_list, preck_avg_list)
        plt.xlabel('Lambda')
        plt.ylabel('Precision@'+str(k))
        plt.title('Precision@' +str(k)+' Achieved at Different Lambda')

        plt.show()

    # Print the best lambda
    if to_print:
        print('The best lambda selected from the cross validation is: ' + str(best_lam))

    # Use the optimal lambda and the entire data set to find the optimal dist
    dist = fit_multivariate_gaussian(data, whitened = True,lam = best_lam)
    return dist

## Support Functions for Performance Evaluation
def eval_prediction(pred,labels,k=10, to_print = False):
    """
    Function to evaluate the correctness of the predictions with multiple metrics
    If rate = True, we will return all the metrics in rate (%) format (except f1)
    """
    results = Results() # Initialize an instance of the Result class

    # Get the indices of the Positive and Negative Predictions
    ind_P = (pred == 1)
    ind_N = (pred == 0)
    results.tp = sum(((pred[ind_P]) == 1) == ((labels[ind_P]) == 1)) # True Positive
    results.tn = sum(((pred[ind_N]) == 0) == ((labels[ind_N]) == 0)) # True Negative
    results.fp = sum(((pred[ind_P]) == 1) == ((labels[ind_P]) == 0)) # False Positive
    results.fn = sum(((pred[ind_N]) == 0) == ((labels[ind_N]) == 1)) # False Negative

    results.Precision = results.tp / max(1,results.tp + results.fp) # Precision Score
    results.Recall = results.tp  / max(1,results.tp + results.fn) # Recall Score
    results.F = (2 * results.Precision * results.Recall) / max(1,results.Precision + results.Recall)
    # A more direct version of f1 is f1 = 2*tp/(2*tp+fn+fp)
    # Find the R-Precision
    results.RPrec,results.R = find_r_prec(pred, labels)
    # Find Precision k - if R < k, we will use R
    results.k = min(k,results.R)
    results.PrecK = find_prec_k(pred, labels,results.k)

    if to_print:
        print("Precision: {0:.1f}%".format(results.Precision * 100))
        print("Recall: {0:.1f}%".format(results.Recall * 100))
        print("F-score: {0:.1f}%".format(results.F * 100))
        print("R-Precision (# R = " + str(results.R) +  "): {0:.1f}%".format(results.RPrec * 100))
        print("Precision@" + str(results.k) +": {0:.1f}%".format(results.PrecK * 100))
        print()
    else:
        return results

def find_r_prec(Preds, Labels):
    """"
    Function to compute R-Precision: average precision for the first n results, where n = # relevant results (anomaly)
    """
    R = sum(Labels) # Total number of relevant data points (anomaly)
    RPrec = find_prec_k(Preds, Labels, R) # Use the find precision-k function to compute R-Precision
    return RPrec, R

def find_prec_k(Preds, Labels,k):
    """
    Compute the Precision at K
    """
    k = int(k) # ensure it is an integer
    PredsK = Preds[0:k] # Prediction at k
    LabelsK = Labels[0:k] # Labels at k
    ind_PK = (PredsK == 1) # Indices of Positive at K
    ind_NK = (PredsK == 0) # Indices of Negative at K
    TPK = np.sum((PredsK[ind_PK] == 1) == (LabelsK[ind_PK] == 1)) # True Positive at K
    FPK = np.sum((PredsK[ind_PK] == 1) == (LabelsK[ind_PK] == 0)) # False Positive at K
    PrecK = TPK/max(1,TPK + FPK) # Precision at K
    return PrecK

def convert_pred(pred,label_Anomaly,label_Normal):
    """
    This function converts the labels in pred into 0 and 1, where 0 indicates Normal, and 1 indicates Anomaly.
    Goal: Convert the data format before comparing to the Labels set for evaluation.
    label_Anomaly: the label used in the input pred set to indicate Anomaly
    label_Normal: the label used in the input pred set to indicate Normal
    """
    pred_output = pred # Copy the Dataset
    pred_output[pred == label_Normal] = 0  # Label the Normality with 0
    pred_output[pred == label_Anomaly] = 1 # Label the Anomalies as 1
    return pred_output


## Support functions for Visualization
def plot_images(imgs,labels):
    """
    To understand the data in image form: Plot 25 images selected randomly and add labels
    """
    ind = np.random.permutation(len(imgs))

    # Create figure with 5x5 sub-plots.
    fig, axes = plt.subplots(5, 5,figsize=(15,15))
    fig.subplots_adjust(hspace=0.1, wspace=0.01)

    for i, ax in enumerate(axes.flat): 
        ax.imshow(imgs[ind[i]], plt.cm.gray)
        if labels[ind[i]] == 1:
            xlabel = 'Anomaly'
        else:
            xlabel = 'Normal'
        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()

def show_anomaly_images(images,labels):
    """
    This function randomly show 9 images with label 1, which is anomaly
    """
    anomaly_label_index = np.asarray(np.where(labels)).reshape(-1) # Get the indice of anomaly
    anomaly_image = [images[i] for i in anomaly_label_index] # Extract the images labeled as anomaly
    anomaly_label = [labels[i] for i in anomaly_label_index] # Extract the images labeled as anomaly
    plot_images(anomaly_image,anomaly_label) # Show 9 images randomly

   
def plot_compare_after_reconst(img_matrix_reconst,imgs_matrix,height,width):
    """
    This function compares the images reconstructed after encoding & decoding with their original one.
    The shape of both image matrice in the input is m*n, where n is the number of components, 
    and m is the number of images.
    """
    # Permutate through the image index
    ind = np.random.permutation(imgs_matrix.shape[0])

    # Create figure with multiple sub-plots.
    fig, axes = plt.subplots(4, 4,figsize=(15,15))
    fig.subplots_adjust(hspace=0.1, wspace=0.01)

    # Initialize the counter of images
    image_count = 0 

    for i, ax in enumerate(axes.flat): 
        if i % 2 == 0:
            image_count += 1
            ax.imshow(imgs_matrix[ind[i],:].reshape(height,width), plt.cm.gray)
            xlabel = "Example {0}: Original Image".format(image_count)
        else:
            ax.imshow(img_matrix_reconst[ind[i-1],:].reshape(height,width), plt.cm.gray)
            xlabel = "Example {0}: Reconstructed Image".format(image_count)
        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

def plot_scatter_with_labels(dist,labels,plot_y_label):
    '''
    This function generates a scatter plot with labels to evaluate the detector
    '''
    # Sort the Images and Labels based on the Probability
    rank = np.argsort(-dist) # Sort from the Smallest to the Largest

    gaps_ranked = dist[rank]
    labels_ranked = labels[rank]


    length = labels_ranked.shape[0]
    counts = list(range(1,length+1))
    colors = labels_ranked == 0

    plt.figure(figsize=(15,8))
    plt.suptitle('Scatter Plot of the ' + plot_y_label,fontsize = 20)
    plt.xlabel('Ranking (ascending)',fontsize = 18)
    plt.ylabel(plot_y_label,fontsize = 18)
    plt.scatter(counts,gaps_ranked,c = colors,cmap=plt.cm.copper) 
    plt.ylim(min(gaps_ranked), max(gaps_ranked))
    plt.show()

def plot_matrix_data(matrix):
    """
    This function plots the distribution of data within a matrix for the purpose of observation
    """
    vector = matrix.flatten() # Convert to a Vector
    rank = np.argsort(vector) # Sort from the Smallest to the Largest
    vector_ranked = vector[rank]
    plt.figure(figsize=(15,8))
    plt.plot(vector_ranked)
    plt.title('The Distribution of Data within the Matrix',fontsize = 20)
    plt.xlabel('Ranking',fontsize = 18)
    plt.ylabel('Data Point Value',fontsize = 18)
    plt.show()

def label_anomaly(labels_input, anomaly_digit):
    """
    This function create a label vector to indicate anomaly. 
    input:
    - labels_input: the input labels vector that contains number 0-9 from MNIST
    - anomaly_digit: the target digit that we define as anomaly
    """
    labels_anomaly = np.zeros(labels_input.shape) # create a zero vector of the same length with the input label vector
    labels_anomaly[labels_input == anomaly_digit] = 1 # Mark the label of the anomaly digit as 1
    return labels_anomaly # return the newly created vector

def scatter_plot_anomaly(data, labels,title = ''):
    """
    Creat a scatter plot of a 2D data contains anomaly
    """
    # print(int(labels[:20]))
    # plt.scatter(data[:,0],data[:,1])
    plt.scatter(data[:,0],data[:,1],c = labels)
    if len(title) > 0:
        plt.title(title)
    plt.show()

def plot_data_2d(data, labels):
    """
    This function creates a 2D Visualization of the input dataset and color with labels.
    Here I use PCA to downsize the multivariate input data into 2-Dimensions.
    Note: the input data has a shape of m*n, where m is the sample size and n is # of dimensions
    """
    n_components = 2
    # Compute PCA with training dataset
    data_encoded,n,m = pca_all_processes(data,labels,n_components,decode = False)
    
    # Print the % variance achieved with 2 PC
    #compare_var(data,data_encoded, to_print = True)

    num_data = min(len(data),4000) # We plot in maximum 4000 points
    data_subset = data[:num_data]
    # Create a Scatterplot of the entire encoded data
    scatter_plot_anomaly(data_encoded, labels,'Scatterplot of the entire dataset')
    # Create multiple scatterplots of the subsets of the encoded data
    plot_data_subsets_2d(data_encoded,labels)
    
def plot_data_2d_autoencoder(AnomalyData,data, labels):
    """
    This function encode the data and plot the 2D representation of the encoded data with PCA
    """
    

    # Specify the model config
    data_dimensions=data.shape[1] # No.dimensions in the data
    encoder_layers_size, decoder_layers_size = get_deep_model_config(data_dimensions,AnomalyData.n_layers,AnomalyData.multiplier)
    # Extract the saved autoencoder model
    autoencoder, encoder = compile_autoencoder(data_dimensions,encoder_layers_size, decoder_layers_size) 
    autoencoder = load_model(AnomalyData.model_path) # Load the saved model
    # Extract the encoder model from the autoencoder model
    encoder_n_layers = len(encoder_layers_size) # Get the number of layers in the encoder
    weights_encoder = autoencoder.get_weights()[0:encoder_n_layers+1] # The first half of the autoencoder model is an encoder model
    encoder.set_weights(weights_encoder) # Set weights

    # Encode the data in the training and the testing set
    data_encoded = encode_data(encoder, data)

    # Anomaly Detection with the Gaussian Model: need to whiten the covariance
    plot_data_2d(data_encoded, labels)
    
def plot_data_subsets_2d(data, labels):
    """
    This function takes a few subsets of data and creates scatterplots of each of them
    
    """
    # Shuffle the index
    ind = np.hstack(range(len(labels)))
    shuffle(ind)
    data_shuffled = data[ind]
    labels_shuffled = labels[ind]
    
    step_size = 500  # Number of points contained in each plot
    
    # Create figure with 5x5 sub-plots.
    fig, axes = plt.subplots(3, 3,figsize=(15,15))
    fig.subplots_adjust(hspace=0.1, wspace=0.01)

    for i, ax in enumerate(axes.flat): 
        start = i*step_size
        end = (i+1)*step_size
        data_subset = data_shuffled[start:end,:] # Get a subset of data with size 500 
        labels_subset = labels_shuffled[start:end] # Get the corresponding labels 
        ax.scatter(data_subset[:,0],data_subset[:,1],c = labels_subset)
        ax.set_title('Scatterplot of ' + str(step_size) + " sample points (No." + str(i+1)+")")
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()        

def plot_heatmap_of_cov(data):
    """
    This function plots a heatmap of the covariance matrix of the input data
    Note: the data is of the size m*n, where m is the sample size and n is the number of dimensions
    """
    print('Description of the data: ')
    print('# Data Points: ' + str(data.shape[0]))
    print('# Dimensions: ' + str(data.shape[1]))
    data_cov = np.cov(data,rowvar=0) # Compute the covariance of each dimensions across rows
    plot_heatmap(data_cov,'Heatmap of the covariance of the entire matrix')
    # plot_heatmap_of_cov_by_segments(data)
    

def plot_heatmap_of_cov_by_segments(data):
    """
    This function aims to plots multiple heatmaps of the covariance of the input data by segments.
    It selects a few columns each time, compute the covariance, and plot the heatmap.
    Note: the data is of the size m*n, where m is the sample size and n is the number of dimensions
    """
    n_dimensions = data.shape[1] # Number of dimensions = # columns
    for i in range(0, n_dimensions, 20):
        end = min(i+20,n_dimensions) # We cannot extract more dimensions than the total number
        data_cov_seg = np.cov(data[i:end,i:end],rowvar = 0)
        subtitle = 'Heatmap of covariance: Dimensions ' + str(i) + ' to '+ str(end)
        plot_heatmap(data_cov_seg,subtitle)
        
    
def plot_heatmap(data,title = ''):
    """
    This function plots a heatmap with the input data matrix 
    """
    plt.imshow(data, cmap='jet', interpolation='nearest') # Create a heatmap
    plt.colorbar() # Add a Color Bar by the side
    if len(title) > 0:
        plt.title(title)
    plt.show()

def plot_2datasets(data1,data2,title1 = '',title2 = ''):
    """
    This function plots the heatmaps of two input data matrix side by side 
    """
    plt.figure(figsize=(15,8))
    plt.subplot(1,2,1)
    plt.imshow(data1, cmap='jet', interpolation='nearest') # Create a heatmap
    if len(title1) > 0:
        plt.title(title1)
    plt.colorbar() # Add a Color Bar by the side

    plt.subplot(1,2,2)
    plt.imshow(data2, cmap='jet', interpolation='nearest') # Create a heatmap
    if len(title2) > 0:
        plt.title(title2)
    plt.colorbar() # Add a Color Bar by the side
    plt.show()
    
def compare_whiten_cov(cov,cov_whitened):
    """
    Plot the heatmatp of the covariance matrix vs. the whitened covariance side by side for comparison
    """
    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.imshow(cov, cmap='jet', interpolation='nearest') # Create a heatmap
    plt.title('Plot of the original Covariance Matrix')
    plt.colorbar() # Add a Color Bar by the side

    plt.subplot(1,2,2)
    plt.imshow(cov_whitened, cmap='jet', interpolation='nearest') # Create a heatmap
    plt.title('Plot of the whitened Covariance Matrix')
    plt.colorbar() # Add a Color Bar by the side

    plt.show()


## Support Function for PCA Methods
def mean_shift(data, component_mean):
    """
    This function applies mean shift to each component in the component matrix
    The input components is a m*n matrix. Each column correspons to one component.
    It is important to return the components' mean vector: we will need it in PCA reconstruction
    """
    data_shifted = (data - component_mean)
    return data_shifted

def plot_eigenfaces(pca_matrix,height, width):
    """
    This function plot the eigenfaces based on the given PCA Matrix
    """
    n_eigen = pca_matrix.shape[1]
    # Define the layout of the plots
    n_row = 4
    n_col = min(5,n_eigen//n_row)

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(n_row, n_col,figsize=(15,15))
    fig.subplots_adjust(hspace=0.1, wspace=0.01)

    for i, ax in enumerate(axes.flat): 
        ax.imshow(pca_matrix[:,i].reshape(height, width), plt.cm.gray)
        xlabel = "Eigenface: {0}".format(i+1)
        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def check_eigen(eigen_value, eigen_vector,cov_matrix):
    """
    This function check the correctness of eigenvector & eigenvalue through the equation
    cov_matrix * eigen_vector = eigen_value * eigen_vector
    """
    for i in range(len(eigen_value)): 
        n = cov_matrix.shape[1]
        eigv = eigen_vector[:,i].reshape(1,n).T 
        np.testing.assert_array_almost_equal(cov_matrix.dot(eigv), eigen_value[i] * eigv, decimal=6, err_msg='', verbose=True)
     
def compute_pca_matrix(data):
    """
    Compute PCA Matrix with the given data
    data: a matrix of size m*n, where m is the number of samples, and n is the # dimensions
    """
    # Record the shape of the data: number of features in columns
    n_features = data.shape[1]

    # Take a mean shift
    component_mean = np.mean(data,axis = 0) # Take mean of each column
    data_shifted = mean_shift(data,component_mean) 

    # compute the covariance matrix of the image matrix
    cov_matrix = np.cov(data_shifted, rowvar=0) # important to add rowvar to specify the axis
    # Compute the eigen value and eigen vectors, where eigenvalue is a vector of length n, and eigenvector is a square matrix of size n*n
    eigen_value, eigen_vector = np.linalg.eig(cov_matrix)

    # Sort the eigenvectors by eigenvalues from large to small 
    # First make a list of (eigenvalue, eigenvector) tuples 
    eig_pairs = [(np.abs(eigen_value[i]), eigen_vector[:,i]) for i in range(len(eigen_value))] 
    # Sort the (eigenvalue, eigenvector) tuples from high to low 
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # Convert the sorted eigen vector list to matrix form
    pca_matrix = np.zeros((n_features,n_features))
    for i in range(0,len(eig_pairs)):
        pca_matrix[:,i] = eig_pairs[i][1]

    # output: 
    # pca_matrix: sorted eigenvectors in a n*n matrix
    # mean: the mean of the original input data (across each dimension)
    return pca_matrix, component_mean

def encode_pca(data, component_mean, pca_matrix, n_components):
    """
    Encode the data with PCA Matrix:
    data: the input data with dimensions m*n
    component_mean: a vector of length n; used for mean shift
    pca_matrix: n*n square matrix
    n_components is the number of components to be used in the PCA Matrix
    """
    # Cut the pca_matrix by columns based on the n_components specified
    pca_matrix_k = pca_matrix[:,:n_components]

    # Take a mean shift on the input data
    data_shifted = mean_shift(data,component_mean) 

    # Compute the encoded image
    # Shape of data_shifted: m * n
    # Shape of pca_matrix: n * k
    # Shape of the output (encoded data): m*k
    data_encoded = data_shifted.dot(pca_matrix_k)
    return data_encoded

def decode_pca(data_encoded, component_mean, pca_matrix, n_components):
    """
    Decode the encoded data with PCA Matrix
    """
    # Cut the pca_matrix by columns based on the n_components specified
    pca_matrix_k = pca_matrix[:,:n_components]

    # Reconstruct through PCA Matrix and Mean Vector
    # Shape of the reconstructed face image matrix: m * n
    data_decoded = data_encoded.dot(pca_matrix_k.T) + component_mean
    return data_decoded

def reconstruct_with_pca(data, component_mean, pca_matrix, n_components):
    """
    Reconstruct the input data with pca: handle both encoding and then decoding
    """
    data_encoded = encode_pca(data, component_mean, pca_matrix, n_components)
    data_decoded = decode_pca(data_encoded, component_mean, pca_matrix, n_components)
    return data_decoded

def pca_all_processes(data,labels,n_components, plot_eigenfaces_bool = False,decode = True, plot_comparison_bool = False, height = 0,width = 0):
    """
    Factorize the process of pca computation and reconstruction in one function
    data: in a matrix form with shape m*n
    labels: a vector where 1 indicates the sample is anomaly, and 0 otherwise
    n_components, number of components after pca encoding
    plot_eigenfaces: trigger to plot the eigenfaces of the image, if True, the height and width of the image should be given
    plot_comparison: trigger to plot the comparison between the original and the reconstructed images; same as above, if true, the height and width of the image should be given
    """
    # Compute PCA Matrix: with the normal data only
    pca_matrix, component_mean = compute_pca_matrix(data[labels == 0])

    if (plot_eigenfaces_bool and height*width !=0): # Plot the eigenfaces only if the data is of the type image
        # Visualize the eigenfaces with the pca matrix
        print("Below is the eigenfaces from the PCA Matrix")
        plot_eigenfaces(pca_matrix[:,:n_components],height, width)
        print()
    
    if decode: # We want to have the decoded data
        # Encode and then decode the entire dataset with the pca_matrix
        data_decoded = reconstruct_with_pca(data, component_mean, pca_matrix, n_components)

        if plot_comparison_bool:
            # Compare the original and reconstructed data in images 
            print("Below is a comparison between the original and the reconstructed data")
            plot_compare_after_reconst(data_decoded,data,height,width) # Function saved in support_functions.py
            print()
        return data_decoded, pca_matrix, component_mean

    else:
        data_encoded = encode_pca(data, component_mean, pca_matrix, n_components)
        return data_encoded, pca_matrix, component_mean

def plot_compare_after_reconst(img_matrix_reconst,imgs_matrix,height,width):
    """
    This function compares the images reconstructed after encoding & decoding with their original one.
    The shape of both image matrice in the input is m*n, where n is the number of components, 
    and m is the number of images.
    """
    if height*width == 0: # Non-image data, plot with heatmap
        plot_2datasets(imgs_matrix[:20],img_matrix_reconst[:20],'Original Data', 'Reconstructed Data') # Plot the top first 20 rows
    else: # Image data, reshape and plot side-by-side
        # Permutate through the image index
        ind = np.random.permutation(imgs_matrix.shape[0])

        # Create figure with multiple sub-plots.
        fig, axes = plt.subplots(4, 4,figsize=(15,15))
        fig.subplots_adjust(hspace=0.1, wspace=0.01)

        # Initialize the counter of images
        image_count = 0 

        for i, ax in enumerate(axes.flat): 
            if i % 2 == 0:
                image_count += 1
                ax.imshow(imgs_matrix[ind[i],:].reshape(height,width), plt.cm.gray)
                xlabel = "Example {0}: Original Image".format(image_count)
            else:
                ax.imshow(img_matrix_reconst[ind[i-1],:].reshape(height,width), plt.cm.gray)
                xlabel = "Example {0}: Reconstructed Image".format(image_count)
            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        plt.show()

def compare_var(data, data_pca,to_print = False):
    '''
    This function compare the % variance achieved with the PCA Encoding 
    '''
    var_retained = np.var(data_pca)/np.var(data)
    if to_print:
        dimension_origin = data.shape[1]
        pca_encoding_dimension = data_pca.shape[1]
        print('Summary of PCA Encoding: ')
        print('Number of Dimension in the Original Dataset: ' + str(dimension_origin))
        print('Number of Dimension in the PCA-Encoded Dataset: '+ str(pca_encoding_dimension))
        print("{0:.1f}% variance is retained with the current PCA Reconstruction.".format(var_retained * 100))
        print()
    return var_retained

def evaludate_pc(data,labels):
    '''
    Evaluate the % variance retained with different number of pc
    '''
    var_retained_list = []
    n_components_list = []
    n_steps = 50 # To save the speed, in maximum we will evaluate 50 PC#s
    step_size = max(int(data.shape[1]/n_steps),1) 
    for n_components in range(0,data.shape[1]+1,step_size):
        data_pca,pca_matrix, component_mean = pca_all_processes(data,labels,n_components)
        var_retained = compare_var(data, data_pca,n_components)
        var_retained_list.append(var_retained)
        n_components_list.append(n_components)
    plt.plot(n_components_list,var_retained_list)
    plt.xlabel('# Components Retained after encoding')
    plt.ylabel('Varaince Retained with PCA Reconstruction')
    plt.title('Evaluation of the number of PC retained in PCA')
    plt.show()



## Support Function for the deep autoencoder
def train_autoencoder(AnomalyData, data, labels,encoder_layers_size,decoder_layers_size,epochs_size = 80, batch_size = 256,dropout =0,save_model = True):
    """
    AnomalyData: an instance of the class Anomaly Data
    data is a matrix of size m*n, where m is the sample size, and n is the dimenions
    labels is a vector of length n
    encoder_layers_size: an array that records the size of each hidden layer in the encoder; if there is only one hidden encoder layer, this will be a numeric value
    decoder_layers_size: an array that records the size of each hidden layer in the decoder; if there is only one hidden decoder layer, this will be a numeric value
    """
    # Generate and Compile a Deep Autoencoder and its encoder
    data_dimensions = data.shape[1] # The dimension = # columns
    autoencoder,encoder = compile_autoencoder(data_dimensions,encoder_layers_size,decoder_layers_size,dropout = dropout)
    # Prepare the input
    # Select only the Normal Image Dataset
    data_normal = data[labels == 0]

    # Split the images and labels
    # By default: 80% in training and 20% in testing
    train_ind, test_ind = perm_and_split(len(data_normal))
    x_train = data_normal[train_ind,:]
    x_test = data_normal[test_ind,:]

    # Normalize the Data
    if AnomalyData.is_image_data:
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
    # Run the model
    autoencoder.fit(x_train, x_train,
                    epochs = epochs_size,
                    batch_size = batch_size,
                    shuffle=True,
                    validation_data=(x_test, x_test)) # x_train images are both the target and input

    # Save and output the model
    if save_model:
        autoencoder.save(AnomalyData.model_path)
    return autoencoder,encoder

def compile_autoencoder(data_length, encoder_layers_size,decoder_layers_size,dropout = 0):
    '''
    Function to construct and compile the deep autoencoder, then return the model
    Input:
        - data_length: size of each data point; used as the height
        - encoder_layers_size,decoder_layers_size: model configuration
    '''
    # Set up the input placeholder
    inputs = Input(shape=(data_length,))

    # Find the number of layers in the encoder and decoder
    n_decoder_layers = len(decoder_layers_size)

    # "encoded" is the encoded representation of the input
    encoded = create_hidden_layers(encoder_layers_size,inputs,dropout = dropout)
    
    # "decoded" is the lossy reconstruction of the input
    decoded = create_hidden_layers(decoder_layers_size,encoded,dropout = dropout)

    # The last output layer: same size as the input
    if dropout>0:
        decoded = Dropout(dropout)(decoded)
    decoded = Dense(data_length, activation='sigmoid')(decoded)
    
    # The autoencoder model maps an input to its reconstruction
    autoencoder = Model(inputs, decoded)
    # The encoder model maps an input to its encoded representation
    encoder = Model(inputs, encoded)
    # Compile the autoencoder
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    
    return autoencoder,encoder

def create_hidden_layers(layers_size, inputs, activation_type = 'relu',dropout = 0):
    """
    This function factorize the creation of hidden layers with Keras.
    """
    model = Dense(int(layers_size[0]),activation = activation_type)(inputs) # The first layer of the model
    n_hidden_layers = len(layers_size) # Find the number of hidden layers in the model (given as an input)
    if n_hidden_layers > 1:
        for i in range(1,n_hidden_layers):
            if dropout> 0:
                model = Dropout(dropout)(model)
            model = Dense(int(layers_size[i]), activation=activation_type)(model)  
    return model

def reconstruct_with_autoencoder(autoencoder,data,visual =False,height = 0, width = 0,image = True):
    """
    Function to reconstruct the data with trained autoencoder
    """
    if image:
        data = data.astype('float32') / 255. # Normalize the Data
    # Load into the model and get the processed output
    data_reconstructed = autoencoder.predict(data)
    if visual:
        # Plot the original images and their reconstructed version for comparison
        print("Below are examples of the Reconstructed Data with Deep Autoencoder")
        plot_compare_after_reconst(data_reconstructed,data,height,width)
    # We returned the data in the end because it is normalized when it is image type
    return data_reconstructed, data

def encode_data(encoder,data,image = True):
    """
    To encode hte data with the trained encoder
    """
    if image:
        data = data.astype('float32') / 255. # Normalize the Data
    # Load into the model and get the processed output
    data_encoded = encoder.predict(data)
    return data_encoded

def build_encoder_layers(n_layers,multiplier,data_dimensions):
    """
    Build layers structure of the encoders
    n_layers: number of layers in the encoders
    multiplier: change factor in layer sizes 
    data_dimensions: # dimensions in the original data
    """
    encoder_layers_size = np.zeros(n_layers) # Initialization
    layer_size = data_dimensions
    for n in range(0,n_layers):
        layer_size = int(layer_size/multiplier) # The new layer has a half-size of the previous layer
        encoder_layers_size[n] = layer_size # Save the layer size
    return encoder_layers_size

def build_decoder_layers(n_layers,multiplier,encoded_dimensions):
    """
    Build layer structure of the decoders
    n_layers: number of layers in the decoders; the last layer will be built based on the input data in the training function
    encoded_dimensions: # dimensions 
    """
    decoder_layers_size = np.zeros(n_layers-1) # Initialization
    layer_size = encoded_dimensions 
    for n in range(0,n_layers - 1):
        layer_size = int(layer_size*multiplier)
        decoder_layers_size[n] = layer_size # save the layer size
    return decoder_layers_size

def set_deep_model_config(input_dimension,n_layers=4,multiplier=2):
    """
    This function set the layers config of encoder and decoder based on the input
    input_dimensions: the dimensions of the input data
    n_layers: number of layers in encoder/decoder
    multiplier: the changing factor in layers (for example, each layer in encoder is half of the size of the previous layer)
    """
    encoder_layers_size = build_encoder_layers(n_layers,multiplier,input_dimension)
    decoder_layers_size = build_decoder_layers(n_layers,multiplier,encoder_layers_size[n_layers-1])
    return encoder_layers_size,decoder_layers_size

def get_deep_model_config(input_dimension,n_layers=4,multiplier=2):
    """
    A function to manage the model configuration: keep consistency so that we only need to tune the model here
    input_dimension: the dimension of the input data
    n_layers: number of layers in encoder/decoder
    multiplier: # each layer in encoder is half of the size of the previous layer
    """
    encoder_layers_size, decoder_layers_size = set_deep_model_config(input_dimension,n_layers=n_layers,multiplier=multiplier)
    return encoder_layers_size, decoder_layers_size

## Other Support Functions
def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step




