function [train_index, test_index] = split(labels,ratio)
%% This function split the images into training and testing with given ratio
% Input:
% images: n*m matrix, where each column represents one sample
% labels: a vector with length m and {0 and 1} as labels
% ratio: a number between [0,1] to represent the precentage of m images to
% be splitted into the training set

m = length(labels); % Sample size
training_size = floor(m*ratio); % Set the size of the training set

while 1
    idx = randperm(m); 
    train_index = idx(1:training_size);
    test_index = idx(training_size+1:end);
    if and(sum(labels(train_index)) > 0, sum(labels(test_index)) >0)
        break % Loop until each set contains at least one label "1"
    end
end
    