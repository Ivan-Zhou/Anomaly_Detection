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
    
    def __init__(self,data_name,folder_path,data_path,n_components,encoder_hidden_layers, decoder_hidden_layers, is_image_data=True,img_height=0,img_width=0,k=20, replicate_for_training = 0,model_path='model_autoencoder.h5'):
        self.data_name = data_name
        self.folder_path = folder_path # String
        self.data_path = folder_path + data_path # String
        self.n_components = n_components # int: No components after PCA encoding
        self.encoder_hidden_layers = encoder_hidden_layers # An array of integers to indicate the structure of encoder network, EXCLUDING THE INPUT LAYER
        self.decoder_hidden_layers = decoder_hidden_layers # An array of integers indicating the structure of the decoder network, EXCLUDING THE OUTPUT LAYER
        self.is_image_data = is_image_data # Boolean: if the data is of the type image
        self.img_height = img_height # int
        self.img_width = img_width # int
        self.k = k # int: a parameter to be used in Precision@k
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
    encoder_hidden_layers = np.array([512,256,128, 64])
    decoder_hidden_layers = np.array([64,128,256,512])
    is_image_data = True
    mnist = AnomalyData(data_name,folder_path,data_path,n_components,encoder_hidden_layers, decoder_hidden_layers, is_image_data=is_image_data)
    return mnist

def set_faces():
    """
    Function to configure MNIST datasets
    """
    data_name = 'Yale Faces'
    folder_path = 'Yale_Faces_Data/'
    data_path = 'CroppedYale/'
    n_components = 50
    encoder_hidden_layers = np.array([252,126,63,31])
    decoder_hidden_layers = np.array([31,62,124,248])
    is_image_data = True
    k = 10
    replicate_for_training = 300
    faces = AnomalyData(data_name,folder_path,data_path,n_components,encoder_hidden_layers, decoder_hidden_layers,is_image_data=is_image_data,k=k,n_layers=n_layers,replicate_for_training=replicate_for_training,multiplier=multiplier)
    return faces

def set_synthetic(folder_path):
    data_name=folder_path
    data_path = 'data/'
    n_components = 14
    encoder_hidden_layers = np.array([13,11,9]) # input dimension is 16
    decoder_hidden_layers = np.array([9,11,13])
    is_image_data = False
    synthetic = AnomalyData(data_name,folder_path,data_path,n_components, encoder_hidden_layers, decoder_hidden_layers,is_image_data=is_image_data,n_layers=n_layers,multiplier=multiplier)
    return synthetic
