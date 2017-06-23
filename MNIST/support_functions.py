import numpy as np
import matplotlib.pyplot as plt  
import random
from random import shuffle

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
        if labels[ind[i]] == 0:
            xlabel = 'Normal'
        else: 
            xlabel = 'Anomaly'
        # xlabel = "Anomaly: {0}".format(int(labels[ind[i]]))
        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()

def plot_eigenfaces(pca_matrix,height, width):
    """
    This function plot the eigenfaces based on the given PCA Matrix
    PCA Matrix: size n*k
    """
    #differfromfaces
    n_eigen = pca_matrix.shape[1]
    # Define the layout of the plots
    n_row = 4
    n_col = min(n_eigen//n_row,5) 

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
    The input components is a m*n matrix. Each row correspons to one component.
    It is important to return the components' mean vector: we will need it in PCA reconstruction
    """
    #differfromface
    component_mean = np.mean(components,axis = 0) # Compute mean of EACH COMPONENT across samples
    shifted_components = (components - component_mean)
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
    The shape of both image matrice in the input is m*n, where n is the number of components, 
    and m is the number of images.
    """
    #differfromface
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
            xlabel = "Example {0}: Reconstructed from PCA".format(image_count)
        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

def pca_reconst(matrix, pca_matrix):
    """
    Apply PCA Encoding and Decoding on the input matrix
    input:
    - matrix: of size m*n
    - pca_matrix: of size n*k
    """
    # Mean Shift
    matrix_shifted, component_mean = mean_shift(matrix)
    
    # Compute the transformed matrix
    # Shape of matrix: m * n
    # Shape of pca_matrix: n * k
    # Shape of the transformed image matrix: m * k
    pca_encoded = matrix_shifted.dot(pca_matrix)

    # Reconstruct through PCA Matrix and Mean Vector
    # Shape of the reconstructed face image matrix: m * n
    pca_decoded = pca_encoded.dot(pca_matrix.T) + component_mean

    return pca_decoded


def find_euclidean_distance(matrix1,matrix2):
    """
    This function find the Euclidean Distance between two Matric
    The distance is between the same columns of two matric
    """
    dist = np.linalg.norm(matrix1 - matrix2,axis = 1) # differfromfaces
    return dist

def select_threshold_distance(edistance, yval,k=10, to_print = False):  
    """
    This function finds the best threshold value to detect the anomaly given the euclidean distance and True label Values
    edistance: euclidean distance 
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

def find_prec_k(Preds, Labels,k):
    """
    Compute the Precision at K
    """
    PredsK = Preds[0:k] # Prediction at k
    LabelsK = Labels[0:k] # Labels at k
    ind_PK = (PredsK == 1) # Indices of Positive at K
    ind_NK = (PredsK == 0) # Indices of Negative at K
    TPK = np.sum((PredsK[ind_PK] == 1) == (LabelsK[ind_PK] == 1)) # True Positive at K
    FPK = np.sum((PredsK[ind_PK] == 1) == (LabelsK[ind_PK] == 0)) # False Positive at K
    PrecK = TPK/max(1,TPK + FPK) # Precision at K
    return PrecK

def eval_with_test(Preds, Labels, k = 10):
    # Find Recall, Precision, F score
    Recall, Precision, F = evaluate_pred(Preds, Labels)
    # Find Precision k
    PrecK = find_prec_k(Preds, Labels,k)
    print("Precision: {0:.1f}%".format(Precision * 100))
    print("Recall: {0:.1f}%".format(Recall * 100))
    print("F-score: {0:.1f}%".format(F * 100))
    print("Precision@" + str(k) +": {0:.1f}%".format(PrecK * 100))

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
