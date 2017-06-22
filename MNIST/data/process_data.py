import pandas as pd
import numpy as np
import cv2

data_path = 'extracted_data/'
save_path = 'input_data/'


def read_imgs(paths, dim = (32,32)):
    '''
    This function read images in the given path and return in a list
    '''
    print("Start Reading Images")
    imgs = []
    for path in paths:
        im = cv2.imread(data_path + path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # Convert from 3-color to gray scale
        resized = cv2.resize(im,dim,interpolation = cv2.INTER_AREA) # Resize the image
        imgs.append(resized)
    imgs = np.asarray(imgs)
    print("Finish Reading Images")
    return imgs


# Read Training Set Labels
data_train = pd.read_csv(data_path + 'train-labels.csv',header =None)
paths_train = data_train[0]
labels_train = data_train[1]

# Read Testing Set Labels
data_test = pd.read_csv(data_path + 'test-labels.csv',header =None)
paths_test = data_test[0]
labels_test = data_test[1]

# Read Training Set Images
imgs_train = read_imgs(paths_train)

# Read Test Set Images
imgs_test = read_imgs(paths_test)

# Save Arraylist
np.save(save_path + 'imgs_train.npy',imgs_train)
np.save(save_path + 'imgs_test.npy',imgs_test)
np.save(save_path + 'labels_train.npy',labels_train)
np.save(save_path + 'labels_test.npy',labels_test)

