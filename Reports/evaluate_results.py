import os, os.path
import scipy.sparse
os.chdir('../')
from AnomalyDataClass import * # Functions to extract parameters of each data files
from support_functions import *

# Set up the test running
read_funcs = [read_mnist_data,
              get_yale_faces_data,
              read_synthetic_data,
              read_synthetic_data,
              read_synthetic_data,
              read_synthetic_data
            ]
parameters = ['',
              '',
              'Synthetic/',
              'Synthetic_2/',
              'Synthetic_3/',
              'Synthetic_4/'
            ]
detect_funcs = [detection_with_pca_reconstruction_error,
                detection_with_pca_gaussian,
                detection_with_autoencoder_reconstruction_error,
                detection_with_autoencoder_gaussian]

n_runs = 10 # Run multiple times and take an average to get a stable result

# Initialize a list to record results
results = []
results.append(n_runs) # Store the number of runs in the first item as a reference

# Loop through read_funcs
for run in range(n_runs): # Run multiple times and get the results
    print('Run #' + str(run+1) + ' starts: ')
    counter = 0 # Reset the counter
    for read_func in read_funcs:
        print(str(read_func))
        if len(parameters[counter]) == 0: # no parameter
            AnomalyData, data_train, data_test, labels_train, labels_test=read_func() # Get the data
        else: 
            AnomalyData, data_train, data_test, labels_train, labels_test=read_func(parameters[counter]) # Get the data
        for detect_func in detect_funcs:
            result = detect_func(AnomalyData,data_train, data_test,labels_train,labels_test,to_print = False)
            results.append(result)
        counter+=1 # Add 1

# Save the data and labels
np.save('evaluation_results.npy', results)
print('Results have been saved!')