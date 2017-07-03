import numpy as np
import random

# Generate 5M numbers, each of which has 16 digits
# Anomaly: number of 1s is less than 4
# Normal: otherwise
np.random.seed(9001)
data = np.random.choice([0,1],size =(10**6,16),p=[0.5,0.5])
data_rowsum = np.sum(data,axis = 1)
labels = data_rowsum < 4
labels = labels*1 # COnvert to 0 and 1
print("Percentage of Anomaly in the dataset: " + str(np.sum(labels)/len(labels)))

np.save('data.npy',data)
np.save('labels.npy',labels)
print('Data and Labels have been saved!')