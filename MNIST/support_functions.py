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
        