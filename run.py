import numpy as np
from support_functions import *
# Give Data
data_train, data_test, labels_train, labels_test = 

# Parameters
dataset_name = '' # String - give a name to indicate the dataset
folder_path = '' # String (Optional) - path to your target folder
data_path = '' # String (Optional) - if you have additional folder for data under the target folder
n_components = 20 # int: Number of components after PCA encoding
encoder_hidden_layers = np.array([512, 256, 128, 64]) # An array of integers to indicate the structure of encoder network, EXCLUDING THE INPUT LAYER (length of the input layer = # dimensions of the input data)
decoder_hidden_layers = np.array([64, 128, 256, 512]) # An array of integers indicating the structure of the decoder network, EXCLUDING THE OUTPUT LAYER 
is_image_data = False # Boolean: if the data is of the type image - for visualization purpose
img_height = 0 # int (optional): height of the image - for visualization purpose
img_width = 0 # int (optional): width of the image - for visualization purpose
k = 10 # int: a parameter to be used in Precision@k
model_path = 'model_autoencoder.h5' # String: the file name of the autoencoder model that you want to save
n_runs = 10 # Run multiple times to get a stable result - all of the results will be saved

# Create an Instance of the Class AnomalyData
Anomaly = AnomalyData(dataset_name, folder_path, data_path, n_components, encoder_hidden_layers. decoder_hidden_layers, is_image_data, img_height, img_width, k, model_path)

#Train the model
data = np.concatenate((data_train, data_test))
labels = np.concatenate((labels_train, labels_test))
autoencoder,encoder = train_autoencoder(Anomaly,data, labels,save_model = True)

# Run detection models
print('Start anomaly detection:')
detect_funcs = [detection_with_pca_reconstruction_error,detection_with_pca_gaussian,detection_with_autoencoder_reconstruction_error,detection_with_autoencoder_gaussian]

# Initialize a list to record results
results = []
results.append(n_runs) # Store the number of runs in the first item as a reference
for run in range(n_runs): # Run multiple times and get the results
    print('Run #' + str(run+1) + ' starts: ')
    for i, detect_func in enumerate(detect_funcs):
        print('Execute Model #' + str(i))
        result = detect_func(Anomaly,data_train, data_test,labels_train,labels_test,to_print = False)
        results.append(result)

# Save the data and labels
np.save('evaluation_results.npy',results)
print('All results have been saved!')



