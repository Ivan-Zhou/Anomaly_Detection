import numpy as np
import random
import matplotlib.pyplot as plt
from random import shuffle

# Generate 100K numbers, each of which has 16 digits
# Anomaly: number of 1s is larger than 2

# Set Parameteres
n_dimensions = 16
n_samples = 10**5
data1_ratio = 0.5 # Dataset 1
data2_ratio = 0.5 # Dataset 2
Anomaly_Threshold = 4 # Anomaly if total # 1s is less than the threshold

def generate_random_mg_data(n_dimensions, n_samples):
    """
    Generate a random dataset 
    """
    mu = np.random.rand(n_dimensions) # Random vector for mean
    cov = np.random.rand(n_dimensions,n_dimensions) # Random matrix for covaraince
    data_mg = np.random.multivariate_normal(mu, cov, size=n_samples) # Generate a random matrix with multivariate normal distribution
    data_bi = data_mg >= 0.5 # Convert to binary - True if the data is larger than 0.5; otherwise 0
    data = data_bi*1 # Convert True/False to 1/0
    return data

def generate_random_data_2md(n_dimensions, data1_size, data2_size):
    """
    Generate a random data set with two multivate gaussian distribution
    """
    # Generate two dataset
    data1 = generate_random_mg_data(n_dimensions, data1_size)
    data2 = generate_random_mg_data(n_dimensions, data2_size)
    # Merge
    data = np.concatenate((data1,data2))

    # Shuffle
    shuffle(data)

    return data

np.random.seed(9001)
# Generate a random data set with two multivate gaussian distribution
data = generate_random_data_2md(n_dimensions,int(n_samples*data1_ratio),int(n_samples*data2_ratio))

# Label Anomaly if the number of 1s is less than 7
data_rowsum = np.sum(data,axis = 1)
labels = data_rowsum < Anomaly_Threshold # Anomaly if total # 1s is less than the threshold
labels = labels*1

print("Percentage of Anomaly in the dataset: " + str(np.sum(labels)/len(labels))) # Find percentage of anomaly in the dataset
print(data[labels == 1][:5]) # Print the first 5 rows of anomaly data as examples

if np.sum(labels)/len(labels) > 0.2:
    print("Too much anomaly: start cleaning!")
    labels_remove = (labels==1) & (np.random.rand(n_samples) <= 0.6) # Remove around 60% of anomalies
    print(str(sum(labels_remove)) +' Anomalies are going to be removed.')
    data = data[~labels_remove] # Remove the selected data
    labels = labels[~labels_remove] # Remove the corresponding labels
    print("Percentage of Anomaly in the dataset after cleaning: " + str(np.sum(labels)/len(labels))) # Find percentage of anomaly in the dataset


# Save the data and labels
np.save('data.npy',data)
np.save('labels.npy',labels)
print('Data and Labels have been saved!')