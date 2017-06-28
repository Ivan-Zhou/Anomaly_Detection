import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from support_functions import *

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

def pca_all_processes(data,labels,n_components,plot_eigenfaces_bool = False,plot_comparison_bool = False, height = 0,width = 0):
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

    if plot_eigenfaces_bool:
        # Visualize the eigenfaces with the pca matrix
        print("Below is the eigenfaces from the PCA Matrix")
        plot_eigenfaces(pca_matrix[:,:n_components],height, width)
    
    # Encode and then decode the entire dataset with the pca_matrix
    data_decoded = reconstruct_with_pca(data, component_mean, pca_matrix, n_components)

    if plot_comparison_bool:
        # Compare the original and reconstructed data in images 
        print("Below is a comparison between the original and the reconstructed data")
        plot_compare_after_reconst(data_decoded,data,height,width) # Function saved in support_functions.py

    return data_decoded, pca_matrix, component_mean