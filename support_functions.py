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

def plot_images(imgs,labels):
    """
    Plot 25 images selected randomly
    """
    ind = np.random.permutation(len(imgs))

    # Create figure with 5x5 sub-plots.
    fig, axes = plt.subplots(5, 5,figsize=(15,15))
    fig.subplots_adjust(hspace=0.1, wspace=0.01)

    for i, ax in enumerate(axes.flat): 
        ax.imshow(imgs[ind[i]], plt.cm.gray)
        xlabel = "Anomaly: {0}".format(labels[ind[i]])
        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()

def show_anomaly_images(images,labels):
    """
    This function randomly show 9 images with label 1
    """
    anomaly_label_index = np.asarray(np.where(labels)).reshape(-1) # Get the indice of anomaly
    anomaly_image = [images[i] for i in anomaly_label_index] # Extract the images labeled as anomaly
    anomaly_label = [labels[i] for i in anomaly_label_index] # Extract the images labeled as anomaly
    plot_images(anomaly_image,anomaly_label) # Show 9 images randomly
    
def plot_eigenfaces(pca_matrix,height, width):
    """
    This function plot the eigenfaces based on the given PCA Matrix
    """
    n_eigen = pca_matrix.shape[1]
    # Define the layout of the plots
    n_row = 4
    n_col = n_eigen//n_row 

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

def mean_shift(components):
    """
    This function applies mean shift to each component in the component matrix
    The input components is a n*m matrix. Each row correspons to one component.
    It is important to return the components' mean vector: we will need it in PCA reconstruction
    """
    component_mean = np.mean(components,axis = 1)
    shifted_components = (components.T - component_mean).T # Necessary to take transpose twice here
    return shifted_components, component_mean

def check_eigen(eigen_value, eigen_vector,cov_matrix):
    """
    This function check the correctness of eigenvector & eigenvalue through the equation
    cov_matrix * eigen_vector = eigen_value * eigen_vector
    """
    for i in range(len(eigen_value)): 
        n = cov_matrix.shape[1]
        eigv = eigen_vector[:,i].reshape(1,n).T 
        np.testing.assert_array_almost_equal(cov_matrix.dot(eigv), eigen_value[i] * eigv, decimal=6, err_msg='', verbose=True)
        
def plot_compare_after_reconst(img_matrix_reconst,imgs_matrix,height,width):
    """
    This function compares the images reconstructed after PCA with their original one.
    The shape of both image matrice in the input is n*m, where n is the number of components, 
    and m is the number of images.
    """
    # Permutate through the image index
    ind = np.random.permutation(imgs_matrix.shape[1])

    # Create figure with multiple sub-plots.
    fig, axes = plt.subplots(4, 4,figsize=(15,15))
    fig.subplots_adjust(hspace=0.1, wspace=0.01)

    # Initialize the counter of images
    image_count = 0 

    for i, ax in enumerate(axes.flat): 
        if i % 2 == 0:
            image_count += 1
            ax.imshow(imgs_matrix[:,ind[i]].reshape(height,width), plt.cm.gray)
            xlabel = "Example {0}: Original Image".format(image_count)
        else:
            ax.imshow(img_matrix_reconst[:,ind[i-1]].reshape(height,width), plt.cm.gray)
            xlabel = "Example {0}: Reconstructed from PCA".format(image_count)
        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

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

def split_train_eval_test(labels,ratio_train = 0.8, ratio_eval = 0):
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
    eval_size = int(m*ratio_eval)
    train_ind = ind_normal[:training_size-1] # Split the Training Set
    nontraining_ind = np.concatenate((ind_normal[training_size:],ind_anomal),axis = 0) # Merge the remaining data
    shuffle(nontraining_ind) # Shuffle the indice of the nontraining set to mix the normal and anomalous dataset
    eval_ind = nontraining_ind[:eval_size-1] # Split the Evaluation Set
    test_ind = nontraining_ind[eval_size:] # Split the Testing Set
    
    if ratio_eval> 0:
        return train_ind, eval_ind, test_ind
    else:
        return train_ind, test_ind

def estimate_gaussian(X):
    """
    Compute the parameters of the Gaussian Distribution
    Note: X is given in the shape of m*k, where k is the number of (reduced) dimensions, and m is the number of images
    """
    mu =np.mean(X,axis=0)
    cov = np.cov(X,rowvar=0)

    return mu, cov

def fit_multivariate_gaussian(data):
    """
    This function is used to compute the mu and cov based on the given data, and fit a multivariate gaussian dist
    This data is given as a m*k matrix, where m represents the number of samples, and k represents the number of dimensions
    """
    mu, cov = estimate_gaussian(data)
    dist = multivariate_normal(mean = mu, cov = cov)
    return dist

def select_threshold(pval, yval, p_anomaly_switch = 0):  
    """
    This function finds the best threshold value to detect the anomaly given the PDF values and True label Values
    pval: Probability based on the Multivariate Gaussian Distribution
    yval: True label value
    p_anomaly_switch: 1 if the given pval is the probability of being an anomaly, otherwise that is the probability
    of not being an anmaly. By default it is 0
    """
    best_epsilon = 0
    best_f1 = 0
    best_tp = 0
    best_fp = 0
    best_fn = 0

    step = (pval.max() - pval.min()) / 1000

    for epsilon in np.arange(pval.min(), pval.max(), step):
        # If the given p is the probability of being an anomaly, we need to find the threshold so that
        # if p > threshold, it will be an anomaly
        # IF the given p is hte probability of not being an anomaly, we need to find the threshold so that 
        # if p < threshold, it will be an anomaly
        if p_anomaly_switch == 1:
            preds = pval > epsilon
        else:
            preds = pval < epsilon

        tp,tn,fp,fn,f1 = eval_prediction(preds,yval)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon
            best_tp = tp
            best_fp = fp
            best_fn = fn
    #return best_epsilon, best_f1, best_tp, best_fp, best_fn
    return best_epsilon

def eval_prediction(pred,yval,rate = False):
    """
    Function to evaluate the correctness of the predictions with multiple metrics
    If rate = True, we will return all the metrics in rate (%) format (except f1)
    """
    true_positive = np.sum(np.logical_and(pred == 1, yval == 1)).astype(float) # True Positive
    true_negative = np.sum(np.logical_and(pred == 0, yval == 0)).astype(float) # True Negative
    false_positive = np.sum(np.logical_and(pred == 1, yval == 0)).astype(float) # False Positive
    false_negative = np.sum(np.logical_and(pred == 0, yval == 1)).astype(float) # False Negative

    precision = true_positive / max(1,true_positive + false_positive)
    recall = true_positive / max(1,true_positive + false_negative)
    f1 = (2 * precision * recall) / max(1,precision + recall)
    # A more direct version of f1 is f1 = 2*tp/(2*tp+fn+fp)
    
    if rate:
        n_p = sum(yval == 1)     # Number of Positive
        n_n = yval.shape[0] - n_p # Number of Negative
        tpr = true_positive/n_p
        tnr = true_negative/n_n
        fpr = false_positive/n_n
        fnr = false_negative/n_p
        return tpr,tnr,fpr,fnr,f1
    else:
        return true_positive,true_negative,false_positive,false_negative,f1
    
def compute_pca_matrix(data, n_components,height, width):
    """
    This function compute the pca matrix with the given data
    The data should be given in the matrix form n*m, where n is the number of dimensions, and m is the number of samples
    The data should have been processed with mean-shift
    n_components: Number of components to be kept after PCA
    """
    # Compute the Covariance Matrix
    cov_matrix = np.cov(data)

    # Compute the eigen value and eigen vectors
    eigen_value, eigen_vector = np.linalg.eig(cov_matrix)

    # Sort the eigenvectors by decreasing eigenvalues
    # First make a list of (eigenvalue, eigenvector) tuples 
    eig_pairs = [(np.abs(eigen_value[i]), eigen_vector[:,i]) for i in range(len(eigen_value))] 
    # Sort the (eigenvalue, eigenvector) tuples from high to low 
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # Convert the sorted eigen vector list to matrix form
    eigen_vector_sorted = np.zeros((height*width,n_components))
    for i in range(0,n_components):
        eigen_vector_sorted[:,i] = eig_pairs[i][1]

    # Cut the sorted eigenvectors by columns to get the transformational matrix for PCA
    pca_matrix = eigen_vector_sorted[:,:n_components]

    return pca_matrix

def find_euclidean_distance(matrix1,matrix2):
    """
    This function find the Euclidean Distance between two Matric
    The distance is between the same columns of two matric
    """
    dist = np.linalg.norm(matrix1 - matrix2,axis = 0) # By specifying axis = 0, we find the distance between columns
    return dist

def select_threshold_distance(edistance, yval):  
    """
    This function finds the best threshold value to detect the anomaly given the PDF values and True label Values
    edistance: euclidean distance 
    yval: True label value
    """
    best_epsilon = 0
    best_f1 = 0
    best_tpr = 0
    best_tnr = 0
    best_fpr = 0
    best_fnr = 0

    step = (edistance.max() - edistance.min()) / 1000

    for epsilon in np.arange(edistance.min(), edistance.max(), step):
        preds = edistance > epsilon
        
        tpr,tnr,fpr,fnr,f1 = eval_prediction(preds,yval,rate = True)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon
            best_tpr = tpr
            best_tnr = tnr
            best_fpr = fpr
            best_fnr = fnr
    return best_epsilon,best_tpr,best_tnr,best_fpr,best_fnr,best_f1
