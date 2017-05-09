%% This function convert all the raw images into gray color scale and cropped out the non-face part
% Enter the folder
cd faces

% Get the x-y coordinates of the faces in each image
load('ImageData.mat')
% Get a list of all jpg files in the directory
imagefiles = dir('*.jpg');
nfiles = length(imagefiles);

%% Read through each image in the folder and process them
for ii = 1:nfiles
    % Read the image
    img_name = imagefiles(ii).name;
    im = imread(img_name);
    % Convert to gray from RGB
    img_gray = rgb2gray(im);
    
    % Read the coordinates of the face in the image
    xy = SubDir_Data(:,ii); % Each column represents an image
    x = xy([1 3 5 7]);
    y = xy([2 4 6 8]);
    % Cropp the image to get the face
    leftx = min(x); % The left boundary
    rightx = max(x); % The right boundary
    bomy = min(y); % Bottom Most
    upy = max(y); % Up Most
    cropped = imcrop(img_gray,[leftx,bomy,rightx,upy]); % Crop the image
    % Save the cropped image
    crop_name = sprintf('cropped/cropped_%d.jpg',ii);
    imwrite(cropped,crop_name);
end    
%% Go back to the home directory
cd ..