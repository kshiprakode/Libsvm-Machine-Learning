clear
load MNIST_digit_data;

rand('seed', 1);
% Loading train data
inds = randperm(size(images_train, 1));
images_train = images_train(inds, :);
labels_train = labels_train(inds, :);
images_train = images_train(1:1000, :);
labels_train = labels_train(1:1000, :);

% Loading test data
inds = randperm(size(images_test, 1));
images_test = images_test(inds, :);
labels_test = labels_test(inds, :);

% Homework 4, Solution 3
% Training the model using svmtrain
model = svmtrain(labels_train, images_train,'-q heart_beat');

% -------------------------------Solution 3---------------------------
disp('---------------------------Solution 3---------------------------');
fprintf('Accuracy (svmpredict) - ');
% Calculating predictions for the test data
[predicted_label, accuracy, decision_values] = svmpredict(labels_test,images_test, model);

miscount = 0;

% Calculating accuracy manually by comparing the predicted labels and the actual labels
for i = 1:size(predicted_label,1)
    if(predicted_label(i) ~= labels_test(i))
        miscount = miscount + 1;
    end
end

Accuracy_Calculated = 1 - miscount/size(predicted_label,1);
fprintf('Calculated Accuracy - %f\n', Accuracy_Calculated);

% -------------------------------Solution 4---------------------------
disp('---------------------------Solution 4---------------------------');

clear
load MNIST_digit_data;

rand('seed', 1);
% Loading train data
inds = randperm(size(images_train, 1));
images_train = images_train(inds, :);
labels_train = labels_train(inds, :);
images_train = images_train(1:1000, :);
labels_train = labels_train(1:1000, :);

% Loading test data
inds = randperm(size(images_test, 1));
images_test = images_test(inds, :);
labels_test = labels_test(inds, :);

% Centering of data
images_train_mc = images_train - mean(images_train);

dimensions = logspace(log10(1),log10(500),50);
mean_square_error = [];

for i = dimensions
    [U, S, V] = svds(images_train_mc,round(i));
    images_train_reduced = images_train_mc * V;
    images_train_reduced = images_train_reduced * V';
    mean_square_error = [mean_square_error immse(images_train_reduced,images_train_mc)];
end

disp('Displaying Eigen Vectors');
% Displaying Eigen Vectors
figure();
V=V';
for i = 1 : 10 
    subplot(2,5,i);
    imagesc(reshape(V(i, :), [28 28]));
end

figure();
plot(dimensions,mean_square_error);
title('Mean Square Error Vs Dimensions');
ylabel('Mean Square Error');
xlabel('Dimensions');

% -------------------------------Solution 5---------------------------

clear
load MNIST_digit_data;

rand('seed', 1);
% Loading train data
inds = randperm(size(images_train, 1));
images_train = images_train(inds, :);
labels_train = labels_train(inds, :);
images_train = images_train(1:1000, :);
labels_train = labels_train(1:1000, :);

% Loading test data
inds = randperm(size(images_test, 1));
images_test = images_test(inds, :);
labels_test = labels_test(inds, :);

images_train_mc = images_train - mean(images_train);
images_test_mc = images_test - mean(images_test);

disp('---------------------------Solution 5---------------------------');

% Finding the SVD for reduced dimensions of 50
[U, S, V] = svds(images_train_mc,50);
images_train_reduced = images_train_mc * V;
images_test_reduced = images_test_mc * V;

disp('Accuracy when the dimensions are reduced to 50');
model = svmtrain(labels_train, images_train_reduced,'-q heart_scale');
[predicted_label, accuracy, decision_values] = svmpredict(labels_test,images_test_reduced, model);

%--------------------------------Solution 6---------------------------
disp('---------------------------Solution 6---------------------------');

clear
load MNIST_digit_data;

rand('seed', 1);

% Loading train data
inds = randperm(size(images_train, 1));
images_train = images_train(inds, :);
labels_train = labels_train(inds, :);
images_train = images_train(1:1000, :);
labels_train = labels_train(1:1000, :);

% Loading test data
inds = randperm(size(images_test, 1));
images_test = images_test(inds, :);
labels_test = labels_test(inds, :);

images_train_mc = images_train - mean(images_train);

images_test_mc = images_test - mean(images_test);

dimensions = [2, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 748];

accuracy_vec = [];
for i = dimensions
    [U, S, V] = svds(images_train_mc,i);
    images_train_reduced = images_train_mc * V;

    images_test_reduced = images_test_mc * V;

    model = svmtrain(labels_train, images_train_reduced,'-q heart_scale');
    [predicted_label, accuracy, decision_values] = svmpredict(labels_test,images_test_reduced, model);

    accuracy_vec = [accuracy_vec accuracy]; 
end


figure();
plot(dimensions,accuracy_vec(1,:));
title('Accuracy Vs Dimensions');
ylabel('Accuracy');
xlabel('Dimensions');


% ----------------------------------Solution 7a ----------------------
clear
load MNIST_digit_data;

rand('seed', 1);

% Loading train data
inds = randperm(size(images_train, 1));
images_train = images_train(inds, :);
labels_train = labels_train(inds, :);
images_train = images_train(1:1000, :);
labels_train = labels_train(1:1000, :);

% Loading test data
inds = randperm(size(images_test, 1));
images_test = images_test(inds, :);
labels_test = labels_test(inds, :);

inputs = images_train;
targets = zeros(size(labels_train,1),10);
labels_train(labels_train == 0) = 10;
for k = 1:1000
    targets(k,labels_train(k)) = 1;
end

inputs = inputs';
targets = targets';

xTest = images_test';
labels_test(labels_test == 0) = 10;
score_vec = [];
net = patternnet(400);
net = train(net,inputs,targets);

tstOutputs = net(xTest);
[~, p] = max(tstOutputs);
score = sum(labels_test' == p);
score = score / size(labels_test,1);
score_vec = [score_vec score];
fprintf('\n');
disp('----------------------------------Solution 7a ----------------------');
fprintf('Accuracy with original train data with hidden neurons - 400.\nAccuracy - %f',score);


% ----------------------------------Solution 7b ----------------------
clear
load MNIST_digit_data;

rand('seed', 1);

% Loading train data
inds = randperm(size(images_train, 1));
images_train = images_train(inds, :);
labels_train = labels_train(inds, :);
images_train = images_train(1:1000, :);
labels_train = labels_train(1:1000, :);

% Loading test data
inds = randperm(size(images_test, 1));
images_test = images_test(inds, :);
labels_test = labels_test(inds, :);

% Mean Centering the data
images_train_mc = images_train - mean(images_train);
images_test_mc = images_test - mean(images_test);

[U, S, V] = svds(images_train_mc,50);
images_train_reduced = images_train_mc * V;
images_test_reduced = images_test_mc * V;

inputs = images_train_reduced;
targets = zeros(size(labels_train,1),10);
labels_train(labels_train == 0) = 10;
for k = 1:1000
    targets(k,labels_train(k)) = 1;
end

inputs = inputs';
targets = targets';

xTest = images_test_reduced';
labels_test(labels_test == 0) = 10;

net = patternnet(18);
net = train(net,inputs,targets);

tstOutputs = net(xTest);
[~, p] = max(tstOutputs);
score = sum(labels_test' == p);
score = score / size(labels_test,1);
fprintf('\n');
disp('----------------------------------Solution 7b ----------------------');
fprintf('Accuracy with 50D train data with hidden neurons - 18\nAccuracy - %f',score);


% ----------------------------------Solution 7c ----------------------
clear
load MNIST_digit_data;

rand('seed', 1);

% Loading train data
inds = randperm(size(images_train, 1));
images_train = images_train(inds, :);
labels_train = labels_train(inds, :);
images_train = images_train(1:1000, :);
labels_train = labels_train(1:1000, :);

% Loading test data
inds = randperm(size(images_test, 1));
images_test = images_test(inds, :);
labels_test = labels_test(inds, :);

fprintf('\n');
disp('----------------------------------Solution 7c ----------------------');

% Mean Centering the data
images_train_mc = images_train - mean(images_train);
images_test_mc = images_test - mean(images_test);

dimensions = [2, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 748];
score_vec = [];
for i = dimensions

    [U, S, V] = svds(images_train_mc,i);
    images_train_reduced = images_train_mc * V;
    images_test_reduced = images_test_mc * V;

    inputs = images_train_reduced;
    targets = zeros(size(labels_train,1),10);
    labels_train(labels_train == 0) = 10;
    for k = 1:1000
        targets(k,labels_train(k)) = 1;
    end
    inputs = inputs';
    targets = targets';

    xTest = images_test_reduced';
    labels_test(labels_test == 0) = 10;
    
    hiddenLayerSize = 100;
    net = patternnet(round(hiddenLayerSize));
    
    net = train(net,inputs,targets);

    tstOutputs = net(xTest);
    [~, p] = max(tstOutputs);
    score = sum(labels_test' == p);
    score = score / size(labels_test,1);
    score_vec = [score_vec score];
    
end
fprintf('Plotting graph for dimensions vs accuracy (Neural Net)\n');
figure();    
plot(dimensions,score_vec);
title('Accuracy Vs Dimensions (Neural Net)');
ylabel('Accuracy');
xlabel('Dimensions');
