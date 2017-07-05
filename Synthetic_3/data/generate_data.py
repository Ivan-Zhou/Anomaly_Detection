import numpy as np
import random
import matplotlib.pyplot as plt
from random import shuffle

# Generate 100K numbers, each of which has 16 digits
# Anomaly: number of 1s is larger than 2

# Set Parameteres
normal_per = 0.9
anomaly_per1 = 0.05 # Anomaly group 1
anomaly_per2 = 0.05 # Anomaly group 2
n_dimensions = 16
n_samples = 10**5

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


np.random.seed(9001)

# Generate three dataset
data_normal = generate_random_mg_data(n_dimensions, int(n_samples*normal_per))
data_anomaly1 = generate_random_mg_data(n_dimensions, int(n_samples*anomaly_per1))
data_anomaly2 = generate_random_mg_data(n_dimensions, int(n_samples*anomaly_per2))

labels_normal = np.zeros(len(data_normal))
labels_anomaly = np.ones(len(data_anomaly1)+ len(data_anomaly2))

# Merge
data_merged = np.concatenate((data_normal,data_anomaly1,data_anomaly2))
labels_merged = np.concatenate((labels_normal,labels_anomaly)) 
print('Shape of the data: ')
print(data_merged.shape)
print('Shape of the labels: ')
print(labels_merged.shape)

# Shuffle
ind = np.hstack(range(len(data_merged)))
shuffle(ind)
data = data_merged[ind]
labels = labels_merged[ind]

# Save the data and labels
np.save('data.npy',data)
np.save('labels.npy',labels)
print('Data and Labels have been saved!')