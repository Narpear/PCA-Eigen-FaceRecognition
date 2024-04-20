close all;
clear all;
load faceData.mat; 
%this data contains 68 images for 10 people 
 
%constructing the training set as the first 64 images of each of the 10 people (1 person = 1 class)
 
train = []; 
for i=1:10; 
    for j=1:64; 
        temp = faceData{i}{j}; 
        temp = temp(:); 
        train = horzcat(train,temp); 
    end;
end;

%Visualize 8 faces of each person to get an idea of the data
count = 1;
figure(1); 
for j = 1:8:640; 
        subplot(10,8,count); imshow(uint8(reshape(train(:,j),64,64))); %see the 10x5 faces
        count = count +1; 
end 
        
meanImg = mean(train,2);
figure; imshow(uint8(reshape(meanImg,64,64))); 
hold on; title('mean face'); 

%prepare covariance matrix 
 
covMatrix = (train-meanImg)*((train-meanImg)');
[eigVec, eigVal] = eig(covMatrix); 
 
%Select appropriate number of (>=90% contribution) eigen vectors 
 
%An example of computing the contribution of the first five eigen values 
contributn = (eigVal(1,1) + eigVal(2,2)+eigVal(3,3) + eigVal(4,4) + eigVal(5,5))/(trace(eigVal))
 
%compute the projection matrix for the reduced eigen space 
redEig = eigVec(:,1:5); 

%visualize the eigen faces; here is an example 
for i=1:5
    figure; imagesc(reshape(redEig(:,i),64,64)); colormap gray;  
    hold on; 
    str1=sprintf('eigen face %d, contribution = %f',i,(eigVal(i,i))/trace(eigVal));
    title(str1);
end

%extract the remaining 4 faces of each person 
test = [];
for i=1:10;
for j=65:68;
    temp = faceData{i}{j};
    temp = temp(:);
    test = horzcat(test,temp); 
end
end 
 
redTestFaces = (test')*redEig; 
redTrainFaces = (train')*redEig;

%Use the minimum distance classifier to label the test samples

% Calculate the Euclidean distance between each test face and the mean face
distances = zeros(size(redTestFaces, 1), 10); % Initialize a matrix to store distances
for i = 1:10
    meanFace = mean(redTrainFaces(i, :), 2); % Calculate the mean face for each person
    distances(:, i) = sqrt(sum((redTestFaces - meanFace).^2, 2)); % Calculate Euclidean distance
end

% Find the class with the smallest distance for each test face
[~, classIndices] = min(distances, [], 2); % Get the indices of the minimum distances

% We evaluate the classifier
% accuracy = sum(classIndices == trueClasses) / length(trueClasses);

% Display the results
disp('Classification results:');
for i = 1:size(classIndices, 1)
    fprintf('Test face %d is classified as person %d.\n', i, classIndices(i));
end

