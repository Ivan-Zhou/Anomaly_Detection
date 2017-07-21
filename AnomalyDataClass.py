class AnomalyData:
    """
    Class for Anomaly Dataset that stores all the parameters to be extracted and used in functions
    Parameters:
    - data_name: the name of the dataset
    - folder_path: the path to access the folder of the data
    - data_path: the path to access the data in the folder
    - model_path: the path to read the deep autoencoder model in the folder
    - is_image_data: indicator if the data is of the type image (If so, it requires normalization before passing to the Deep Autoencoder)
    - n_components: number of components remained after PCA encoding
    - k: used to compute the precision@k
    - n_layers: # layers in deep autoencoder model
    - replicate_for_training: If the dataset is too small, we will replicate the data before training the deep autoencodering
    - multiplier: by what rate should the new layer in the encoder model should decreasing than the previous layer
    """
    
    def __init__(self,data_name,folder_path,data_path,n_components,is_image_data=True,img_height=0,img_width=0,k=20,n_layers=4,multiplier=2,replicate_for_training = 0,model_path='model_autoencoder.h5'):
        self.data_name = data_name
        self.folder_path = folder_path # String
        self.data_path = folder_path + data_path # String
        self.n_components = n_components # int: No components after PCA encoding
        self.is_image_data = is_image_data # Boolean: if the data is of the type image
        self.img_height = img_height # int
        self.img_width = img_width # int
        self.k = k # int: a parameter to be used in Precision@k
        self.n_layers = n_layers # int: no of layers
        self.multiplier = multiplier # double: #neurons in new layer/#neuron in previous layer in encoder model
        self.replicate_for_training = replicate_for_training # Integer
        self.model_path = folder_path + model_path # String

def set_mnist():
    """
    Function to configure MNIST datasets
    """
    data_name = 'MNIST'
    folder_path = 'MNIST/'
    data_path = 'data/'
    n_components = 200
    is_image_data = True
    n_layers = 4
    multiplier = 2
    mnist = AnomalyData(data_name,folder_path,data_path,n_components,is_image_data=is_image_data,n_layers=n_layers,multiplier=multiplier)
    return mnist

def set_faces():
    """
    Function to configure MNIST datasets
    """
    data_name = 'Yale Faces'
    folder_path = 'Yale_Faces_Data/'
    data_path = 'CroppedYale/'
    n_components = 50
    is_image_data = True
    k = 10
    n_layers = 4
    multiplier = 2
    replicate_for_training = 300
    faces = AnomalyData(data_name,folder_path,data_path,n_components,is_image_data=is_image_data,k=k,n_layers=n_layers,replicate_for_training=replicate_for_training,multiplier=multiplier)
    return faces

def set_synthetic(folder_path):
    data_name=folder_path
    data_path = 'data/'
    n_components = 14
    is_image_data = False
    n_layers = 3
    multiplier = 1.2
    synthetic = AnomalyData(data_name,folder_path,data_path,n_components,is_image_data=is_image_data,n_layers=n_layers,multiplier=multiplier)
    return synthetic
