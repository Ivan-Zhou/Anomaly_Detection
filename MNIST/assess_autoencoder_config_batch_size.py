import numpy as np 
import matplotlib.pyplot as plt
from processing_functions import *
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from support_functions import *
from Autoencoder_Functions import *
import time

# Configure the Parameters:
config_target = 'Batch Size' # Text to be printed in the final graph
n_layers = 5 # number of layers in decoder and encoder
size_multiplier = 2 # Multiplier of change in layer size: must be larger than 1
batchSize_multiplier = 2 # Multiplier of change in batch size: must be larger than 1
epochsSize = 70 # For the training of the deep autoencoder
k = 50 # Metric parameter: For the Precision k

    
## Pre-Process Data
# Read data
data_path = 'data/input_data/'
anomaly_digit = 2
# Read image matrix (n*m), labels (vector of m), and image size
imgs_train, imgs_test, labels_train, labels_test, height, width = read_process_data(data_path, anomaly_digit)
# The length of one image vector
img_size = height*width 
# Merge the data
imgs = np.concatenate((imgs_train, imgs_test))
labels = np.concatenate((labels_train, labels_test))

# Initialization
n_config_test = 4
RPrec_list = np.zeros(n_config_test)
Preck_list = np.zeros(n_config_test)
batchSize_list = np.zeros(n_config_test)
time_training_list = np.zeros(n_config_test)
count = 0
batchSize = 256 # Initialization

for i in range(0,n_config_test):
    # Model Config
    encoder_layers_size = build_encoder_layers(n_layers,size_multiplier,img_size)
    decoder_layers_size = build_decoder_layers(n_layers,size_multiplier,encoder_layers_size[n_layers-1])
    
    time_start = time.time()

    # Training
    autoencoder,encoder = train_autoencoder(imgs, labels,encoder_layers_size,decoder_layers_size,epochs_size = epochsSize,batch_size = batchSize,save_model = False)
    # Notification
    print('Config #' + str(count+1) + ': Finish training!')

    time_training = time.time() - time_start
    
    # Evaluating the detector with the Reconstruction Error
    data_train_decoded,data_train_normal = reconstruct_with_autoencoder(autoencoder,imgs_train,visual = False)
    data_test_decoded,data_test_normal = reconstruct_with_autoencoder(autoencoder,imgs_test,visual=False)
    Recall,Precision,F,RPrec,R,PrecK = train_test_with_reconstruction_error(data_train_normal, data_train_decoded, data_test_normal, data_test_decoded, labels_train, labels_test,k,to_print = False)
    # Record the result
    RPrec_list[count] = RPrec
    Preck_list[count] = PrecK
    time_training_list[count] = time_training
    batchSize_list[count] = batchSize

    # Notification
    print('Config #' + str(count+1) + ': Finish test!')
    count +=1
    batchSize = batchSize*batchSize_multiplier # Increase the batchSize

# Plot Config vs. Target Measurement
print('Create a plot for the result.')
plt.figure(figsize=(15,8))
plt.subplot(1,3,1)
plt.plot(batchSize_list, RPrec_list)
plt.xlabel(config_target)
plt.ylabel('R-Precision (@R = '+str(R))
plt.title('Metric # 1: R-Precision')

plt.subplot(1,3,2)
plt.plot(batchSize_list, Preck_list)
plt.xlabel(config_target)
plt.ylabel('Precision @ '+str(k))
plt.title('Metric # 2: Precision @ ' + str(k))

plt.subplot(1,3,3)
plt.plot(batchSize_list, time_training_list)
plt.xlabel(config_target)
plt.ylabel('Training Time')
plt.title('Metric # 3: Training Time (s)')

plt.savefig('results/autoencoder_config_assessment_batch_size.png')
print('Plot is saved.')
print('The assessment is finished!')





