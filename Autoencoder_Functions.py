import numpy as np  
import pandas as pd  

from support_functions import *

from keras.layers import Input, Dense
from keras.models import Model

def train_autoencoder(data, labels,encoder_layers_size,decoder_layers_size,epochs_size = 80, batch_size = 256,save_model = True):
    """
    data is a matrix of size m*n, where m is the sample size, and n is the dimenions
    labels is a vector of length n
    encoder_layers_size: an array that records the size of each hidden layer in the encoder; if there is only one hidden encoder layer, this will be a numeric value
    decoder_layers_size: an array that records the size of each hidden layer in the decoder; if there is only one hidden decoder layer, this will be a numeric value
    """
    # Generate and Compile a Deep Autoencoder and its encoder
    data_dimensions = data.shape[1] # The dimension = # columns
    autoencoder,encoder = compile_autoencoder(data_dimensions,encoder_layers_size,decoder_layers_size)

    # Prepare the input
    # Select only the Normal Image Dataset
    data_normal = data[labels == 0]
    # Split the images and labels
    # By default: 80% in training and 20% in testing
    train_ind, test_ind = perm_and_split(len(data_normal))
    x_train = data_normal[train_ind,:]
    x_test = data_normal[test_ind,:]

    # Normalize the Data
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    # x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    # x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

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

        
def compile_autoencoder(data_length, encoder_layers_size,decoder_layers_size):
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
    encoded = create_hidden_layers(encoder_layers_size,inputs)
    
    # "decoded" is the lossy reconstruction of the input
    decoded = create_hidden_layers(decoder_layers_size,encoded)

    # The last output layer: same size as the input
    decoded = Dense(data_length, activation='sigmoid')(decoded)
    
    # The autoencoder model maps an input to its reconstruction
    autoencoder = Model(inputs, decoded)
    # The encoder model maps an input to its encoded representation
    encoder = Model(inputs, encoded)
    # Compile the autoencoder
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    
    return autoencoder,encoder

def create_hidden_layers(layers_size, inputs, activation_type = 'relu'):
    """
    This function factorize the creation of hidden layers with Keras. 
    The size of 
    """
    model = Dense(layers_size[0],activation = activation_type)(inputs) # The first layer of the model
    n_hidden_layers = len(layers_size) # Find the number of hidden layers in the model (given as an input)
    if n_hidden_layers > 1:
        for i in range(1,n_hidden_layers):
            model = Dense(layers_size[i], activation=activation_type)(model)  
    return model

def reconstruct_with_autoencoder(autoencoder,data,visual =False,height = 0, width = 0):
    """
    Function to reconstruct the data with trained autoencoder
    """
    data_normal = data.astype('float32') / 255. # Normalize the Data
    # Load into the model and get the processed output
    data_reconstructed = autoencoder.predict(data_normal)
    if visual:
        # Plot the original images and their reconstructed version for comparison
        plot_compare_after_reconst(data_reconstructed,data,height,width)
    return data_reconstructed

def encode_data(encoder,data):
    """
    To encode hte data with the trained encoder
    """
    data_normal = data.astype('float32') / 255. # Normalize the Data
    # Load into the model and get the processed output
    data_encoded = encoder.predict(data_normal)
    return data_encoded