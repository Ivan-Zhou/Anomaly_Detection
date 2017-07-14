import os
import numpy as np
from support_functions import *

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

def faces_autoencoder_reconstruction_error(k=10, data_path='Yale_Faces_Data/CroppedYale/',model_path = 'Yale_Faces_Data/model_autoencoder.h5',to_print=False,n_layers=4,multiplier=2,is_image_data=True):
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
        detection_with_autoencoder_reconstruction_error(imgs_train, imgs_test,labels_train,labels_test,k,model_path,is_image_data=is_image_data,to_print = to_print,n_layers=n_layers,multiplier=multiplier,height=height,width=width)
    else:
        Recall,Precision,F,RPrec,R,PrecK = detection_with_autoencoder_reconstruction_error(imgs_train, imgs_test,labels_train,labels_test,k,model_path,is_image_data=is_image_data,to_print = to_print,n_layers=n_layers,multiplier=multiplier)
        return Recall,Precision,F,RPrec,R,PrecK

def faces_autoencoder_gaussian(k=10, data_path='Yale_Faces_Data/CroppedYale/',model_path = 'Yale_Faces_Data/model_autoencoder.h5',to_print=False,n_layers=4,multiplier=2,is_image_data=True):
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
        detection_with_autoencoder_gaussian(imgs_train, imgs_test,labels_train,labels_test,k,model_path,is_image_data=is_image_data,to_print = to_print,n_layers=n_layers,multiplier=multiplier,height=height,width=width)
    else:
        Recall,Precision,F,RPrec,R,PrecK = detection_with_autoencoder_gaussian(imgs_train, imgs_test,labels_train,labels_test,k,model_path,is_image_data=is_image_data,to_print = to_print,n_layers=n_layers,multiplier=multiplier)
        return Recall,Precision,F,RPrec,R,PrecK

def synthetic_pca_reconstruction_error(folder_path, n_components=14,k=50,to_print=False):
    """
    Run anomaly detection with PCA and Reconstruction Error on any of the synthetic Data (in the specified folder)
    Parameters:
    - folder_path: the path lead to the right folder (as this function applies to all the synthetic dataset)
    - n_components: number of components remained after PCA encoding
    - k: used to compute the precision@k
    """
    # Read image matrix (n*m), labels (vector of m), and image size
    data, labels, data_train, data_test, labels_train, labels_test = read_synthetic_data(folder_path)

    # Anomaly Detection with Reconstruction Error
    if to_print:
        detection_with_pca_reconstruction_error(data_train, data_test,labels_train,labels_test,n_components,k,to_print = to_print)
    else: 
        Recall,Precision,F,RPrec,R,PrecK = detection_with_pca_reconstruction_error(data_train, data_test,labels_train,labels_test,n_components,k,to_print = to_print)
        return Recall,Precision,F,RPrec,R,PrecK 

def synthetic_pca_gaussian(folder_path, n_components=14,k=50,to_print=False):
    """
    Run anomaly detection with PCA and Gaussian on any of the synthetic Data (in the specified folder)
    Parameters:
    - folder_path: the path lead to the right folder (as this function applies to all the synthetic dataset)
    - n_components: number of components remained after PCA encoding
    - k: used to compute the precision@k
    """
    # Read image matrix (n*m), labels (vector of m), and image size
    data, labels, data_train, data_test, labels_train, labels_test = read_synthetic_data(folder_path)

    # Anomaly Detection with Reconstruction Error
    if to_print:
        detection_with_pca_gaussian(data_train, data_test,labels_train,labels_test,n_components,k,to_print = to_print)
    else: 
        Recall,Precision,F,RPrec,R,PrecK = detection_with_pca_gaussian(data_train, data_test,labels_train,labels_test,n_components,k,to_print = to_print)
        return Recall,Precision,F,RPrec,R,PrecK 

def synthetic_autoencoder_reconstruction_error(folder_path, k=50,model_path='model_autoencoder.h5',to_print=False,n_layers=2,multiplier=2,is_image_data=False):
    """
    Run anomaly detection with Deep Autoencoder and Reconstruction Error on any of the synthetic Data (in the specified folder)
    Parameters:
    - folder_path: the path lead to the right folder (as this function applies to all the synthetic dataset)
    - k: used to compute the precision@k
    - is_image_data: indicate if the data is of image type - will be normalized if true
    """
    # Read image matrix (n*m), labels (vector of m), and image size
    data, labels, data_train, data_test, labels_train, labels_test = read_synthetic_data(folder_path)

    # Anomaly Detection with Reconstruction Error
    if to_print: 
        detection_with_autoencoder_reconstruction_error(data_train, data_test,labels_train,labels_test,k,folder_path+model_path,is_image_data=is_image_data,to_print = to_print,n_layers=n_layers,multiplier=multiplier)
    else:
        Recall,Precision,F,RPrec,R,PrecK = detection_with_autoencoder_reconstruction_error(data_train, data_test,labels_train,labels_test,k,folder_path+model_path,is_image_data=is_image_data,to_print = to_print,n_layers=n_layers,multiplier=multiplier)
        return Recall,Precision,F,RPrec,R,PrecK

def synthetic_autoencoder_gaussian(folder_path, k=50,model_path='model_autoencoder.h5',to_print=False,n_layers=2,multiplier=1.5,is_image_data=False):
    """
    Run anomaly detection with Deep Autoencoder and Gaussian Method on any of the synthetic Data (in the specified folder)
    Parameters:
    - folder_path: the path lead to the right folder (as this function applies to all the synthetic dataset)
    - k: used to compute the precision@k
    - is_image_data: indicate if the data is of image type - will be normalized if true
    """
    # Read image matrix (n*m), labels (vector of m), and image size
    data, labels, data_train, data_test, labels_train, labels_test = read_synthetic_data(folder_path)

    # Anomaly Detection with Reconstruction Error
    if to_print: 
        detection_with_autoencoder_gaussian(data_train, data_test,labels_train,labels_test,k,folder_path+model_path,is_image_data=is_image_data,to_print = to_print,n_layers=n_layers,multiplier=multiplier)
    else:
        Recall,Precision,F,RPrec,R,PrecK = detection_with_autoencoder_gaussian(data_train, data_test,labels_train,labels_test,k,folder_path+model_path,is_image_data=is_image_data,to_print = to_print,n_layers=n_layers,multiplier=multiplier)
        return Recall,Precision,F,RPrec,R,PrecK