class AnomalyData:
    """
    Class for Anomaly Dataset that stores all the parameters to be extracted and used in functions
    Parameters:
    - folder_path: the path to access the folder of the data
    - data_path: the path to access the data in the folder
    - model_path: the path to read the deep autoencoder model in the folder
    - is_image_data: indicator if the data is of the type image (If so, it requires normalization before passing to the Deep Autoencoder)
    - n_components: number of components remained after PCA encoding
    - k: used to compute the precision@k
    - n_layers: # layers in deep autoencoder model
    - multiplier: by what rate should the new layer in the encoder model should decreasing than the previous layer
    """
    
    def __init__(self,folder_path,data_path,n_components,is_image_data=True,img_height=0,img_width=0,k=20,n_layers=4,multiplier=2,model_path='model_autoencoder.h5'):
        self.folder_path = folder_path # String
        self.data_path = folder_path + data_path # String
        self.n_components = n_components # int: No components after PCA encoding
        self.is_image_data = is_image_data # Boolean: if the data is of the type image
        self.img_height = img_height # int
        self.img_width = img_width # int
        self.k = k # int: a parameter to be used in Precision@k
        self.n_layers = n_layers # int: no of layers
        self.multiplier = multiplier # double: #neurons in new layer/#neuron in previous layer in encoder model
        self.model_path = folder_path + model_path # String

def set_mnist():
    """
    Function to configure MNIST datasets
    """
    folder_path = 'MNIST/'
    data_path = 'data/'
    n_components = 200
    is_image_data = True
    n_layers = 4
    multiplier = 2
    mnist = AnomalyData(folder_path,data_path,n_components,is_image_data=is_image_data,n_layers=n_layers,multiplier=multiplier)
    return mnist

def set_faces():
    """
    Function to configure MNIST datasets
    """
    folder_path = 'Yale_Faces_Data/'
    data_path = 'data/'
    n_components = 50
    is_image_data = True
    k = 10
    n_layers = 4
    multiplier = 2
    faces = AnomalyData(folder_path,data_path,n_components,is_image_data=is_image_data,k=k,n_layers=n_layers,multiplier=multiplier)
    return faces

def set_synthetic(folder_path):
    data_path = 'data/'
    n_components = 14
    is_image_data = False
    n_layers = 2
    multiplier = 1.5
    synthetic = AnomalyData(folder_path,data_path,n_components,is_image_data=is_image_data,n_layers=n_layers,multiplier=multiplier)
