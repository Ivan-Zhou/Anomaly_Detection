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
from keras.models import load_model
from sklearn.model_selection import KFold


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
    # Find the R-Precision
    RPrec,R = find_r_prec(pred, yval)
    # Find Precision k - if R < k, we will use R
    PrecK = find_prec_k(pred, yval,min(k,R))
    # A more direct version of f1 is f1 = 2*tp/(2*tp+fn+fp)
    
    if rate:
        n_p = sum(yval == 1)     # Number of Positive
        n_n = yval.shape[0] - n_p # Number of Negative
        tpr = true_positive/n_p
        tnr = true_negative/n_n
        fpr = false_positive/n_n
        fnr = false_negative/n_p
        return tpr,tnr,fpr,fnr,f1,RPrec,PrecK
    else:
        return true_positive,true_negative,false_positive,false_negative,f1,RPrec,PrecK

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

    # Sort the edistance and yval based on pval from high to low (in order to measure the Precision at K)
    rank = np.argsort(-edistance) # Sort from the Largest to the Smallest
    # If we want a rank from the smallest to the largest, we need to change the label here
    dist_ranked = edistance[rank] # Sort the edistance
    yval_ranked = yval[rank] # Sort the yval with the same order

    # Step Size
    step = (dist_ranked.max() - dist_ranked.min()) / 100

    for epsilon in np.arange(dist_ranked.min(), dist_ranked.max(), step):
        preds_ranked = dist_ranked > epsilon # If the distance is larger than the threshold, it will be identified as an anomaly

        tp,tn,fp,fn,f1,RPrec,precK = eval_prediction(preds_ranked,yval_ranked,k,rate = True)

        # Optimize to find the highest precision at k
        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    # Get the best measurement with the best threshold
    if to_print: # Print out the result
        best_preds = dist_ranked > best_epsilon # If the pval is larger than the threshold, it will be identified as an anomaly
        eval_with_test(best_preds, yval_ranked, k) # Print out the result

    return best_epsilon

def select_threshold_probability(p, yval, k=10, to_print = False):  
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

    # Sort the edistance and yval based on pval from high to low (in order to measure the Precision at K)
    rank = np.argsort(p) # Sort from the smallest to the largest
    # If we want a rank from the smallest to the largest, we need to change the label here
    p_ranked = p[rank] # Sort the Probability
    yval_ranked = yval[rank] # Sort the yval with the same order

    # Step Size
    if p_ranked.max() == p_ranked.min(): # A horizontal line: No need to find epsilon
        best_epsilon = 0
        best_f1 = 0
    else:
        step = (p_ranked.max() - p_ranked.min()) / 100

        for epsilon in np.arange(p_ranked.min(), p_ranked.max(), step):
            preds_ranked = p_ranked < epsilon # If the probability is smaller than the threshold, it will be identified as an anomaly

            tp,tn,fp,fn,f1,RPrec,precK = eval_prediction(preds_ranked,yval_ranked,k,rate = True)

            # Optimize to find the highest precision at k
            if f1 > best_f1:
                best_f1 = f1
                best_epsilon = epsilon

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

def eval_with_test(Preds, Labels, k = 10,to_print = True):
    """
    Function to print out the metrices (for the evaluation on both the training and testing dataset)
    """
    # Find Recall, Precision, F score
    Recall, Precision, F = evaluate_pred(Preds, Labels)
    # Find the R-Precision
    RPrec,R = find_r_prec(Preds, Labels)
    # Find Precision k
    PrecK = find_prec_k(Preds, Labels,k)
    if to_print:
        print("Precision: {0:.1f}%".format(Precision * 100))
        print("Recall: {0:.1f}%".format(Recall * 100))
        print("F-score: {0:.1f}%".format(F * 100))
        print("R-Precision (# R = " + str(R) +  "): {0:.1f}%".format(RPrec * 100))
        print("Precision@" + str(k) +": {0:.1f}%".format(PrecK * 100))
        print()
    else:
        return Recall,Precision,F,RPrec,R,PrecK

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

    # Get the number of actual anomaly for the R-precision
    r_train = sum(labels_train)
    # Train the Anomaly Detector
    if to_print:
        print("Training Results:")
        threshold_error = select_threshold_distance(dist_train, labels_train,r_train,k,to_print = to_print)
        print()
    else: # no print
        threshold_error = select_threshold_distance(dist_train, labels_train,r_train,k,to_print = to_print)

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
        eval_with_test(preds, labels_test_ranked, k,to_print = to_print)
    else: # no print & with return
        Recall,Precision,F,RPrec,R,PrecK = eval_with_test(preds, labels_test_ranked, k,to_print = to_print)
        return Recall,Precision,F,RPrec,R,PrecK

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
        eval_with_test(preds, labels_test_ranked, k,to_print = to_print)
    else: # no print & with return
        Recall,Precision,F,RPrec,R,PrecK = eval_with_test(preds, labels_test_ranked, k,to_print = to_print)
        return Recall,Precision,F,RPrec,R,PrecK

def fit_gaussian_with_whiten_and_cv(data,labels,folds,k,to_print = True):
    """
    Here we fit a multivariate gaussian with whitening and cross validation
    to_print: if true, plot the comparison between the original and whitened cov
    """
    kf = KFold(n_splits = folds) # Create multiple folds for cross validation (cv)
    best_rprec_avg = 0 # Initialize the best average RPrec 
    best_f1_avg = -1 # Intialize a list to record the best lambda 
    lam_list = [] # list to record the lambda - for the plot
    f1_avg_list = []
    rprec_avg_list = [] # list to record the average RPrec corresponding to each lambda - used for the plot
    preck_avg_list = []
    
    for lam in frange(0,0.999,0.09): # Loop through each possible lambda (discretized)
        f1_list = []
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

            tp,tn,fp,fn,f1,RPrec,precK = eval_prediction(preds,labels_test_ranked,k,rate = True)
            f1_list.append(f1) # Save the f1 score of the current training & testing combination
            preck_list.append(precK)
            rprec_list.append(RPrec)

        f1_avg = sum(f1_list)/len(f1_list) # The average f1 score for the current lambda
        rprec_avg = sum(rprec_list)/len(rprec_list)
        preck_avg = sum(preck_list)/len(preck_list)

        # Save the current lambda and rprec_avg
        lam_list.append(lam)
        f1_avg_list.append(f1_avg)
        rprec_avg_list.append(rprec_avg)
        preck_avg_list.append(preck_avg)

        # Optimize to find the highest f1
        if f1_avg > best_f1_avg:
            best_lam = lam # Record the current lambda
            best_f1_avg = f1_avg # Record the current target measurement
        if to_print:
            print('Finish evaluate Lambda: ' + str(lam))

    if to_print:
        plt.figure(figsize=(15,8))
        plt.subplot(1,2,1)
        plt.plot(lam_list, rprec_avg_list)
        plt.xlabel('Lambda')
        plt.ylabel('R-Precision')
        plt.title('R-Precision Achieved at Different Lambda')

        plt.subplot(1,2,2)
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

def train_autoencoder(data, labels,encoder_layers_size,decoder_layers_size,epochs_size = 80, batch_size = 256,dropout =0,image = True, save_model = True):
    """
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
    if image:
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
        autoencoder.save('model_autoencoder.h5')
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

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

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
    compare_var(data,data_encoded, to_print = True)

    num_data = min(len(data),4000) # We plot in maximum 4000 points
    data_subset = data[:num_data]
    # Create a Scatterplot of the entire encoded data
    scatter_plot_anomaly(data_encoded, labels,'Scatterplot of the entire dataset')
    # Create multiple scatterplots of the subsets of the encoded data
    plot_data_subsets_2d(data_encoded,labels)
    
    
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


def compare_var(data, data_pca,pca_encoding_dimension,to_print = False):
    '''
    This function compare the % variance achieved with the PCA Encoding 
    '''
    var_retained = np.var(data_pca)/np.var(data)
    if to_print:
        dimension_origin = data.shape[1]
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

def detection_with_pca_reconstruction_error(data_train, data_test,labels_train,labels_test,n_components,k,to_print = False,height=0,width=0):
    """
    Function to apply anomaly detection with PCA and Reconstruction Error
    """
    if to_print:
        evaludate_pc(data_train,labels_train) # Evaluate the % variance achieved at different #PC

    # Compute PCA with training dataset, and reconstruct the training dataset
    data_train_pca,pca_matrix,component_mean = pca_all_processes(data_train,labels_train,n_components,plot_eigenfaces_bool=to_print,plot_comparison_bool=to_print,height=height,width=width)
    # Reconstruct the test set
    data_test_pca = reconstruct_with_pca(data_test, component_mean,pca_matrix,n_components)

    if to_print: 
        compare_var(data_train, data_train_pca,n_components,to_print = to_print) # Find the % variance achieved at the current #PC

    # Anomaly Detection with Reconstruction Error
    if to_print: # Print result
        train_test_with_reconstruction_error(data_train, data_train_pca, data_test, data_test_pca, labels_train, labels_test,k,to_print = to_print)
    else:  # Return results in numeric values
        Recall,Precision,F,RPrec,R,PrecK = train_test_with_reconstruction_error(data_train, data_train_pca, data_test, data_test_pca, labels_train, labels_test,k,to_print = to_print)
        return Recall,Precision,F,RPrec,R,PrecK

def detection_with_pca_gaussian(data_train, data_test,labels_train,labels_test,n_components,k,to_print = False,height=0,width=0):
    """
    Function to apply anomaly detection with PCA and Gaussian
    """
    if to_print:
        evaludate_pc(data_train,labels_train) # Evaluate the % variance achieved at different #PC
    # Compute PCA with training dataset and encode the training dataset
    data_train_encoded,pca_matrix, component_mean = pca_all_processes(data_train,labels_train,n_components,plot_eigenfaces_bool = to_print,decode = False,height=height,width=width)
    # Encode the test set
    data_test_encoded = encode_pca(data_test, component_mean,pca_matrix,n_components)

    if to_print: 
        data_train_pca = reconstruct_with_pca(data_train, component_mean, pca_matrix, n_components) # Reconstruct with PCA
        compare_var(data_train, data_train_pca,n_components,to_print = to_print) # FInd the % variance achieved at the current #PC

    # Anomaly Detection with the Gaussian Model
    if to_print: # Print result
        train_test_with_gaussian(data_train_encoded, data_test_encoded, labels_train, labels_test,k,to_print=to_print)
    else:  # Return results in numeric values
        Recall,Precision,F,RPrec,R,PrecK = train_test_with_gaussian(data_train_encoded, data_test_encoded, labels_train, labels_test,k,to_print=to_print)
        return Recall,Precision,F,RPrec,R,PrecK

def detection_with_autoencoder_reconstruction_error(data_train, data_test,labels_train,labels_test,k,model_path,is_image_data=True,to_print = False,n_layers=4,multiplier=2,height=0,width=0):
    """
    Function to apply anomaly detection with Autoencoder and Reconstruction Error
    model_path: path that stores the model
    is_image_data: boolean to indicate if the data is of image type
    """
    # Generate and Compile a Deep Autoencoder
    # Specify the model config
    data_dimensions=data_train.shape[1] # No.dimensions in the data
    encoder_layers_size, decoder_layers_size = get_deep_model_config(data_dimensions,n_layers,multiplier)
    # Extract the saved model
    autoencoder, encoder = compile_autoencoder(data_dimensions,encoder_layers_size, decoder_layers_size) 
    autoencoder = load_model(model_path) # Load the saved model

    # Print the summary  of the autoencoder model
    if to_print:
        print('Below is a summery of the autoencoder model: ')
        print(autoencoder.summary())
        print("\n The output shape of the autoencoder model: ")
        print(autoencoder.output_shape)
    
    # Reconstruct the training data with autoencoder
    data_train_reconstructed,data_train = reconstruct_with_autoencoder(autoencoder,data_train,visual =to_print,height = height, width = width,image=is_image_data)

    # Reconstruct the testing data
    data_test_reconstructed,data_test = reconstruct_with_autoencoder(autoencoder,data_test,visual =False,height = height, width = width,image=is_image_data)

    # Anomaly Detection with Reconstruction Error
    if to_print: # Print result
        train_test_with_reconstruction_error(data_train, data_train_reconstructed, data_test, data_test_reconstructed, labels_train, labels_test,k,to_print = to_print)
    else:  # Return results in numeric values
        Recall,Precision,F,RPrec,R,PrecK = train_test_with_reconstruction_error(data_train, data_train_reconstructed, data_test, data_test_reconstructed, labels_train, labels_test,k,to_print = to_print)
        return Recall,Precision,F,RPrec,R,PrecK

def detection_with_autoencoder_gaussian(data_train, data_test,labels_train,labels_test,k,model_path,is_image_data=True,to_print = False,n_layers=4,multiplier=2,height=0,width=0):
    """
    Function to apply anomaly detection with Autoencoder and Multivariate Gaussian Method
    model_path: path that stores the model
    is_image_data: boolean to indicate if the data is of image type
    """
    # Generate and Compile an encoder
    # Specify the model config
    data_dimensions=data_train.shape[1] # No.dimensions in the data
    encoder_layers_size, decoder_layers_size = get_deep_model_config(data_dimensions,n_layers,multiplier)
    # Extract the saved autoencoder model
    autoencoder, encoder = compile_autoencoder(data_dimensions,encoder_layers_size, decoder_layers_size) 
    autoencoder = load_model(model_path) # Load the saved model
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
        data_train_reconstructed,data_train = reconstruct_with_autoencoder(autoencoder,data_train,visual =to_print,height = height, width = width,image=is_image_data)
    # Encode the data in the training and the testing set
    data_train_encoded = encode_data(encoder, data_train)
    data_test_encoded = encode_data(encoder, data_test)

    # Anomaly Detection with the Gaussian Model: need to whiten the covariance
    if to_print: # Print result
        train_test_with_gaussian(data_train_encoded, data_test_encoded, labels_train, labels_test,k,whitened = True, plot_comparison = to_print, to_print=to_print)
    else:  # Return results in numeric values
        Recall,Precision,F,RPrec,R,PrecK = train_test_with_gaussian(data_train_encoded, data_test_encoded, labels_train, labels_test,k,whitened = True, plot_comparison = to_print, to_print=to_print)
        return Recall,Precision,F,RPrec,R,PrecK

def read_synthetic_data(folder_path=''):
    """
    Automate the process to read and process the data in any of the synthetic folder
    """
    data_path = 'data/'
    data_fname = 'data.npy'
    labels_fname = 'labels.npy'

    # Load
    data = np.load(folder_path + data_path + data_fname)
    labels = np.load(folder_path + data_path + labels_fname)

    # Split the data and labels into the training & testing groups
    # Split the images and labels
    ratio_train = 0.7 # No training set
    train_ind, test_ind = split_training(labels,ratio_train)

    data_train = data[train_ind] 
    data_test = data[test_ind]
    labels_train = labels[train_ind]
    labels_test = labels[test_ind]

    return data,labels, data_train, data_test, labels_train, labels_test