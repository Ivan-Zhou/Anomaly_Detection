import os
import numpy as np
from support_functions import *
from PCA_Functions import *
from Autoencoder_Functions import *

os.chdir('Yale_Faces_Data')
# print(os.getcwd())
from processing_functions_yale_faces import *
os.chdir('../')

os.chdir('MNIST')
from processing_functions_mnist import * 
os.chdir('../')

def mnist_pca_reconstruction_error(anomaly_digit=2, n_components=200,k=20, data_path='MNIST/data/input_data/',to_print=False):
    """
    Run anomaly detection with PCA and Reconstruction Error on the MNIST Data
    Parameters:
    - anomay_digit: the digit in MNIST that is treated as anomaly
    - n_components: number of components remained after PCA encoding
    - k: used to compute the precision@k
    - data_path: the path to read all MNIST data
    """
    # Read image matrix (n*m), labels (vector of m), and image size
    imgs_train, imgs_test, labels_train, labels_test, height, width = read_process_data(data_path, anomaly_digit)

    # Anomaly Detection with Reconstruction Error
    if to_print:
        detection_with_pca_reconstruction_error(imgs_train, imgs_test,labels_train,labels_test,n_components,k,to_print = to_print,height=height,width=width)
    else: 
        Recall,Precision,F,RPrec,R,PrecK = detection_with_pca_reconstruction_error(imgs_train, imgs_test,labels_train,labels_test,n_components,k,to_print = to_print)
        return Recall,Precision,F,RPrec,R,PrecK 

def mnist_pca_gaussian(anomaly_digit=2, n_components=200,k=20, data_path='MNIST/data/input_data/',to_print=False):
    """
    Run anomaly detection with PCA and Gaussian on the MNIST Data
    Parameters:
    - anomay_digit: the digit in MNIST that is treated as anomaly
    - n_components: number of components remained after PCA encoding
    - k: used to compute the precision@k
    - data_path: the path to read all MNIST data
    """
    # Read image matrix (n*m), labels (vector of m), and image size
    imgs_train, imgs_test, labels_train, labels_test, height, width = read_process_data(data_path, anomaly_digit)

    # Anomaly Detection with Gaussian
    if to_print: 
        detection_with_pca_gaussian(imgs_train,imgs_test,labels_train,labels_test,n_components,k,to_print=to_print,height=height,width=width)
    else:
        Recall,Precision,F,RPrec,R,PrecK = detection_with_pca_gaussian(imgs_train,imgs_test,labels_train,labels_test,n_components,k,to_print=to_print)
        return Recall,Precision,F,RPrec,R,PrecK

def faces_pca_gaussian(n_components=50,k=10, data_path='Yale_Faces_Data/CroppedYale/',to_print=False):
    """
    Run anomaly detection with PCA and Gaussian on the faces Data
    Parameters:
    - n_components: number of components remained after PCA encoding
    - k: used to compute the precision@k
    - data_path: the path to read all Yale Faces data
    """
    # Read image matrix (n*m), labels (vector of m), and image size
    imgs, labels, height, width = get_yale_faces_data(data_path)

    # Split the images and labels
    imgs_train,imgs_test,labels_train,labels_test = split_data_labels_training_testing(imgs,labels)

    # Anomaly Detection with Gaussian
    if to_print: 
        detection_with_pca_gaussian(imgs_train,imgs_test,labels_train,labels_test,n_components,k,to_print=to_print,height=height,width=width)
    else:
        Recall,Precision,F,RPrec,R,PrecK = detection_with_pca_gaussian(imgs_train,imgs_test,labels_train,labels_test,n_components,k,to_print=to_print)
        return Recall,Precision,F,RPrec,R,PrecK

def faces_pca_reconstruction_error(n_components=50,k=10, data_path='Yale_Faces_Data/CroppedYale/',to_print=False):
    """
    Run anomaly detection with PCA and Reconstruction Error on the faces Data
    Parameters:
    - n_components: number of components remained after PCA encoding
    - k: used to compute the precision@k
    - data_path: the path to read all Yale Faces data
    """
    # Read image matrix (n*m), labels (vector of m), and image size
    imgs, labels, height, width = get_yale_faces_data(data_path)

    # Split the images and labels
    imgs_train,imgs_test,labels_train,labels_test = split_data_labels_training_testing(imgs,labels)

    # Anomaly Detection with Reconstruction Error
    if to_print: 
        detection_with_pca_reconstruction_error(imgs_train,imgs_test,labels_train,labels_test,n_components,k,to_print=to_print,height=height,width=width)
    else:
        Recall,Precision,F,RPrec,R,PrecK = detection_with_pca_reconstruction_error(imgs_train,imgs_test,labels_train,labels_test,n_components,k,to_print=to_print)
        return Recall,Precision,F,RPrec,R,PrecK

def mnist_autoencoder_reconstruction_error(anomaly_digit=2,k=50, data_path='MNIST/data/input_data/',model_path='MNIST/model_autoencoder.h5',to_print=False,is_image_data=True):
    """
    Run anomaly detection with Deep Autoencoder and Reconstruction Error on the MNIST Data
    Parameters:
    - anomay_digit: the digit in MNIST that is treated as anomaly
    - k: used to compute the precision@k
    - data_path: the path to read all MNIST data
    """
    # Read image matrix (n*m), labels (vector of m), and image size
    imgs_train, imgs_test, labels_train, labels_test, height, width = read_process_data(data_path, anomaly_digit)

    # Anomaly Detection with Reconstruction Error
    if to_print:
        detection_with_autoencoder_reconstruction_error(imgs_train, imgs_test,labels_train,labels_test,k,model_path,is_image_data=is_image_data,to_print = to_print,height=height,width=width)
    else:
        Recall,Precision,F,RPrec,R,PrecK = detection_with_autoencoder_reconstruction_error(imgs_train, imgs_test,labels_train,labels_test,k,model_path,is_image_data=is_image_data,to_print = to_print)
        return Recall,Precision,F,RPrec,R,PrecK
    
def mnist_autoencoder_gaussian(anomaly_digit=2,k=50, data_path='MNIST/data/input_data/',model_path='MNIST/model_autoencoder.h5',to_print=False,is_image_data=True):
    """
    Run anomaly detection with Deep Autoencoder and Reconstruction Error on the MNIST Data
    Parameters:
    - anomay_digit: the digit in MNIST that is treated as anomaly
    - k: used to compute the precision@k
    - data_path: the path to read all MNIST data
    """
    # Read image matrix (n*m), labels (vector of m), and image size
    imgs_train, imgs_test, labels_train, labels_test, height, width = read_process_data(data_path, anomaly_digit)

    # Anomaly Detection with Multivariate Gaussian
    if to_print:
        detection_with_autoencoder_gaussian(imgs_train, imgs_test,labels_train,labels_test,k,model_path,is_image_data=is_image_data,to_print = to_print,height=height,width=width)
    else:
        Recall,Precision,F,RPrec,R,PrecK = detection_with_autoencoder_gaussian(imgs_train, imgs_test,labels_train,labels_test,k,model_path,is_image_data=is_image_data,to_print = to_print)
        return Recall,Precision,F,RPrec,R,PrecK

def faces_autoencoder_reconstruction_error(k=10, data_path='Yale_Faces_Data/CroppedYale/',model_path = 'Yale_Faces_Data/model_autoencoder.h5',to_print=False,is_image_data=True):
    """
    Run anomaly detection with Deep Autoencoder and Reconstruction Error on the faces Data
    Parameters:
    - k: used to compute the precision@k
    - data_path: the path to read all Yale Faces data
    """
    # Read image matrix (n*m), labels (vector of m), and image size
    imgs, labels, height, width = get_yale_faces_data(data_path)

    # Split the images and labels
    imgs_train,imgs_test,labels_train,labels_test = split_data_labels_training_testing(imgs,labels)

    # Anomaly Detection with Reconstruction Error
    if to_print: 
        detection_with_autoencoder_reconstruction_error(imgs_train, imgs_test,labels_train,labels_test,k,model_path,is_image_data=is_image_data,to_print = to_print,height=height,width=width)
    else:
        Recall,Precision,F,RPrec,R,PrecK = detection_with_autoencoder_reconstruction_error(imgs_train, imgs_test,labels_train,labels_test,k,model_path,is_image_data=is_image_data,to_print = to_print)
        return Recall,Precision,F,RPrec,R,PrecK

def faces_autoencoder_gaussian(k=10, data_path='Yale_Faces_Data/CroppedYale/',model_path = 'Yale_Faces_Data/model_autoencoder.h5',to_print=False,is_image_data=True):
    """
    Run anomaly detection with Deep Autoencoder and Reconstruction Error on the faces Data
    Parameters:
    - k: used to compute the precision@k
    - data_path: the path to read all Yale Faces data
    """
    # Read image matrix (n*m), labels (vector of m), and image size
    imgs, labels, height, width = get_yale_faces_data(data_path)

    # Split the images and labels
    imgs_train,imgs_test,labels_train,labels_test = split_data_labels_training_testing(imgs,labels)

    # Anomaly Detection with Multivariate Gaussian
    if to_print:
        detection_with_autoencoder_gaussian(imgs_train, imgs_test,labels_train,labels_test,k,model_path,is_image_data=is_image_data,to_print = to_print,height=height,width=width)
    else:
        Recall,Precision,F,RPrec,R,PrecK = detection_with_autoencoder_gaussian(imgs_train, imgs_test,labels_train,labels_test,k,model_path,is_image_data=is_image_data,to_print = to_print)
        return Recall,Precision,F,RPrec,R,PrecK