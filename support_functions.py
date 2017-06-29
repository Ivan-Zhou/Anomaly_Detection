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
        
def estimate_gaussian(X):
    """
    Compute the parameters of the Gaussian Distribution
    Note: X is given in the shape of m*k, where k is the number of (reduced) dimensions, and m is the number of images
    """
    mu = np.mean(X,axis=0)
    cov = np.cov(X,rowvar=0)

    return mu, cov

def fit_multivariate_gaussian(data):
    """
    This function is used to compute the mu and cov based on the given data, and fit a multivariate gaussian dist
    This data is given as a m*k matrix, where m represents the number of samples, and k represents the number of dimensions
    """
    mu, cov = estimate_gaussian(data)
    dist = multivariate_normal(mean = mu, cov = cov,allow_singular=False)
    return dist

def eval_prediction(pred,yval,k, rate = False):
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
    PrecK = find_prec_k(pred, yval,k)
    # A more direct version of f1 is f1 = 2*tp/(2*tp+fn+fp)
    
    if rate:
        n_p = sum(yval == 1)     # Number of Positive
        n_n = yval.shape[0] - n_p # Number of Negative
        tpr = true_positive/n_p
        tnr = true_negative/n_n
        fpr = false_positive/n_n
        fnr = false_negative/n_p
        return tpr,tnr,fpr,fnr,f1,PrecK
    else:
        return true_positive,true_negative,false_positive,false_negative,f1,PrecK

def find_euclidean_distance(matrix1,matrix2):
    """
    This function find the Euclidean Distance between two Matric
    The distance is between the same columns of two matric
    """
    dist = np.linalg.norm(matrix1 - matrix2,axis = 1) # By specifying axis = 0, we find the distance between columns
    return dist

def select_threshold_distance(edistance, yval,r, k=10, to_print = False):  
    """
    This function finds the best threshold value to detect the anomaly given the euclidean distance and True label Values
    edistance: euclidean distance 
    yval: True label value
    r: the number of relevant results for R-Precision test
    k: the k used to compute the precision at k in the testing set
    to_print: indicate if the result need to be printed
    """
    # Initialize the Metrics: only the selected will be used in optimization
    best_epsilon = 0
    best_f1 = 0
    # best_tp = 0
    # best_fp = 0
    # best_fn = 0
    # best_precK = 0 # Precision at k

    # Sort the edistance and yval based on pval from high to low (in order to measure the Precision at K)
    rank = np.argsort(-edistance) # Sort from the Largest to the Smallest
    # If we want a rank from the smallest to the largest, we need to change the label here
    dist_ranked = edistance[rank] # Sort the edistance
    yval_ranked = yval[rank] # Sort the yval with the same order

    # Step Size
    step = (dist_ranked.max() - dist_ranked.min()) / 1000

    for epsilon in np.arange(dist_ranked.min(), dist_ranked.max(), step):
        preds_ranked = dist_ranked > epsilon # If the distance is larger than the threshold, it will be identified as an anomaly

        tp,tn,fp,fn,f1,precK = eval_prediction(preds_ranked,yval_ranked,k,rate = True)

        # Optimize to find the highest precision at k
        if f1 > best_f1:
            best_epsilon = epsilon # Record the current epsilon
            best_f1 = f1 # Record the current target measurement: f1

    # Get the best measurement with the best threshold
    if to_print: # Print out the result
        best_preds = dist_ranked > best_epsilon # If the pval is larger than the threshold, it will be identified as an anomaly
        eval_with_test(best_preds, yval_ranked, k) # Print out the result

    return best_epsilon

def select_threshold_probability(p, yval,k=10, to_print = False):  
    """
    This function finds the best threshold value to detect the anomaly given the PDF values and True label Values
    p: probability given by the Multivariate Gaussian Model
    yval: True label value
    k: the k used to compute the precision at k in the testing set
    to_print: indicate if the result need to be printed
    """
    # Initialize the Metrics: only the selected will be used in optimization
    best_epsilon = 0
    best_f1 = 0
    # best_tp = 0
    # best_fp = 0
    # best_fn = 0
    # best_precK = 0 # Precision at k

    # Sort the edistance and yval based on pval from high to low (in order to measure the Precision at K)
    rank = np.argsort(p) # Sort from the smallest to the largest
    # If we want a rank from the smallest to the largest, we need to change the label here
    p_ranked = p[rank] # Sort the Probability
    yval_ranked = yval[rank] # Sort the yval with the same order

    # Step Size
    step = (p_ranked.max() - p_ranked.min()) / 1000

    for epsilon in np.arange(p_ranked.min(), p_ranked.max(), step):
        preds_ranked = p_ranked < epsilon # If the probability is smaller than the threshold, it will be identified as an anomaly

        tp,tn,fp,fn,f1,precK = eval_prediction(preds_ranked,yval_ranked,k,rate = True)

        # Optimize to find the highest precision at k
        if f1 > best_f1:
            best_epsilon = epsilon # Record the current epsilon
            best_f1 = f1 # Record the current target measurement: f1

    # Get the best measurement with the best threshold
    if to_print: # Print out the result
        best_preds = p_ranked < best_epsilon # If the pval is larger than the threshold, it will be identified as an anomaly
        eval_with_test(best_preds, yval_ranked, k) # Print out the result

    return best_epsilon

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


def evaluate_pred(Preds, Labels):
    """
    This function evaluate the Prediction with Labels. 
    Standard Format: Positive (Anomaly) is labeled with 1, and Negative (Normal) is labeled with 0
    """
    # Get the indices of the Positive and Negative Predictions
    ind_P = (Preds == 1)
    ind_N = (Preds == 0)
    # Evaluation
    TP = sum(((Preds[ind_P]) == 1) == ((Labels[ind_P]) == 1)) # True Positive
    TN = sum(((Preds[ind_N]) == 0) == ((Labels[ind_N]) == 0)) # True Negative
    FP = sum(((Preds[ind_P]) == 1) == ((Labels[ind_P]) == 0)) # False Positive
    FN = sum(((Preds[ind_N]) == 0) == ((Labels[ind_N]) == 1)) # False Negative
    TP, TN, FP, FN
    # Compute the Precision and Recall
    Recall = TP/max(1,TP+FN)
    Precision = TP/max(1,TP+FP)
    F = (2*Precision*Recall) / max(1,Precision+Recall)
    
    return Recall, Precision, F

def eval_with_test(Preds, Labels, k = 10):
    """
    Function to print out the metrices (for the evaluation on both the training and testing dataset)
    """
    # Find Recall, Precision, F score
    Recall, Precision, F = evaluate_pred(Preds, Labels)
    # Find the R-Precision
    RPrec = find_r_prec(Preds, Labels)
    # Find Precision k
    PrecK = find_prec_k(Preds, Labels,k)
    print("Precision: {0:.1f}%".format(Precision * 100))
    print("Recall: {0:.1f}%".format(Recall * 100))
    print("F-score: {0:.1f}%".format(F * 100))
    print("R-Precision: {0:.1f}%".format(PrecK * 100))
    print("Precision@" + str(k) +": {0:.1f}%".format(PrecK * 100))

def find_r_prec(Preds, Labels):
    """"
    Function to compute R-Precision: average precision for the first n results, where n = # relevant results (anomaly)
    """

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

def train_test_with_reconstruction_error(data_original_train, data_decoded_train, data_original_test, data_decoded_test, labels_train, labels_test,k):
    """
    Factorize the training and testing process of the Reconstruction Error-based method
    """
    ## Training
    # Find the euclidean distance between the original dataset and the decoded dataset
    dist_train = find_euclidean_distance(data_decoded_train,data_original_train)

    # Plot of the reconstruction error from high to low
    print("The higher the reconstruction error, the more likely the point will be an anomaly")
    plot_scatter_with_labels(dist_train,labels_train,"Reconstruction Error")

    # Get the number of actual anomaly for the R-precision
    r_train = sum(labels_train)
    # Train the Anomaly Detector
    print("Training Results:")
    threshold_error = select_threshold_distance(dist_train, labels_train,r_train,k,to_print = True)
    print()

    ## Testing
    # Find the euclidean distance between the original dataset and the decoded dataset
    dist_test = find_euclidean_distance(data_decoded_test,data_original_test)

    # Sort the Images and Labels based on the Reconstruction Error
    rank_test = np.argsort(-dist_test) # Sort from the Largest to the Smallest
    dist_test_ranked = dist_test[rank_test] # Sort the Reconstruction Error
    # Rank Labels accoring to the same order
    labels_test_ranked = labels_test[rank_test]

    # Give Predictions
    preds = np.zeros(labels_test.shape) # Initialization
    preds[dist_test_ranked > threshold_error] = 1

    # Get the number of actual anomaly for the precision-k test
    k_test = sum(labels_test)
    # Evaluate the Detector with Testing Data
    print("Testing Results:")
    eval_with_test(preds, labels_test_ranked, k_test)

def train_test_with_gaussian(data_train, data_test, labels_train, labels_test):
    """
    Factorize the training and testing process of the Multivariate Gaussian-based method
    """
    ## Training
    # Get Gaussian Distribution Model with the Training Data
    # Note: fit_multivariate_gaussian() is my own coded function
    dist = fit_multivariate_gaussian(data_train)

    # Get Probability of being Anomaly vs. being Normal
    p_train = dist.pdf(data_train)   # Probability of Being Normal

    # Plot the Probability with labels
    plot_scatter_with_labels(p_train, labels_train,'Gaussian Probability')

    # Get the number of actual anomaly for the precision-k test
    k_train = sum(labels_train)
    # Train the Anomaly Detector
    print("Training Results:")
    threshold_gaussian  = select_threshold_probability(p_train, labels_train,k_train,to_print = True)
    print()

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

    k_test = sum(labels_test)
    # Evaluate the Detector with Testing Data
    eval_with_test(preds, labels_test_ranked, k_test)