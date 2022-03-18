% lung adenocarcinomas (n = 127)
% Other adenocarcinomas (n = 12) were suspected to be extrapulmonary metastases
% normal lung (n = 17) specimens
% SCLC (n = 6) cases
% squamous cell lung carcinomas (n = 21)
% pulmonary carcinoids (n = 20)

clc; clear all; close all;

% %% Load Gene Expression Data
% numvars = 203;
% type = 'double';
% vartype = repmat({type},1,203);
% opts = spreadsheetImportOptions('NumVariables', numvars, 'VariableTypes', vartype);
% opts.DataRange = 'C2';
% data = readmatrix('dataseta_12600gene.xls', opts);
% X = data';
% 
% % import gene names cells
% clear opts
% opts = spreadsheetImportOptions('VariableNames', 'gene');
% opts.DataRange = 'B2';
% genes = readmatrix('dataseta_12600gene.xls', opts);
% 
% %% Load Classification Types
% % import data into cells
% clear opts
% opts = spreadsheetImportOptions('NumVariables', numvars);
% opts.DataRange = 'C1:GW1';
% labels = readmatrix('dataseta_12600gene.xls', opts);
% 
% % labels for different lung tissue samples
% cancerName = {'lung adenocarcinomas',
%               'normal lung',
%               'SCLC',
%               'squamous cell lung carcinomas',
%               'pulmonary carcinoids'};
% 
% % class numbers for corresponding labels
% class = {1, 2, 3, 4, 5};
% 
% % map the labels to a number
% map = containers.Map(class, cancerName);
% 
% % store class for each corresponding label in variable y
% y = zeros(numvars, 1);
% for n = 1:numvars
%     % 'lung adenocarcinomas', 
%     if (labels{1,n}(1,1:2) == 'AD')
%         y(n) = 1;
%     % 'normal lung',    
%     elseif (labels{1,n}(1,1:2) == 'NL')    
%         y(n) = 2;        
%     % 'SCLC',    
%     elseif (labels{1,n}(1,1:2) == 'SM')    
%         y(n) = 3;    
%     % 'squamous cell lung carcinomas',    
%     elseif (labels{1,n}(1,1:2) == 'SQ')    
%         y(n) = 4;
%     % 'pulmonary carcinoids',    
%     elseif (labels{1,n}(1,1:2) == 'CO')    
%         y(n) = 5;
%     end
% end
% 
% class = cell2mat(class);
% numSamp = size(X, 1);
% numFeat = size(X, 2);
% 

%% Load Data
load('dataLoad.mat')


%% ERGS

% find the indices of which samples belong to which label
classNum = y(1);
count = 1;
classIndicesBegin = zeros(max(y)+1, 1);
classIndicesBegin(1) = 1;
classIndicesEnd = zeros(max(y)+1,1);
for n = 1:numSamp
    if (y(n) ~= classNum)
       classNum = y(n);
       classIndicesBegin(count+1) = n;
       classIndicesEnd(count) = n-1;
       count = count+1;
    end
end
classIndicesEnd(count) = numSamp;

% NOTE: classIndices contains the last index of each class in the y array

% initialize mean, prior probability, and std of each feature in each class
mu_j_c = zeros(length(class),numFeat);
p_c = zeros(length(class), 1);
sigma_j_c = zeros(size(mu_j_c));

% initialize Effective Range Matrices
Rupper = zeros(size(mu_j_c));
Rlower = zeros(size(Rupper));

% take gamma to be 1.732, which says that the fraction of entries of X with
% P(|X-mu_j_c| >= gamma*sigma_j_c) <= 1/(gamma^2), is no more than 1/(gamma^2)
% with gamma > 1.  The effective range includes at least 2/3 (1-1/3) of the
% data objects
prob = 1/3;
gamma = sqrt(1/prob);

% for each class
for c = 1:length(class)
   % find the mean of each feature (j) for each class (c)
   mu_j_c(c,:) = mean(X(classIndicesBegin(c):classIndicesEnd(c), :));
   
   % find the prior probability of each class
   p_c(c) = (classIndicesEnd(c) - classIndicesBegin(c) + 1) / numSamp;
   
   % find the std of each feature (j) for each class (c)
   sigma_j_c(c,:) = std(X(classIndicesBegin(c):classIndicesEnd(c), :));
   
   % calculate the Effective Ranges
   Rupper(c,:) = mu_j_c(c,:) + (1 - p_c(c)).*gamma.*sigma_j_c(c,:);
   Rlower(c,:) = mu_j_c(c,:) - (1 - p_c(c)).*gamma.*sigma_j_c(c,:);

end

% sort the effective ranges of classes in ascending order to compute 
% Overlapping Area (OAi) for each feature Xi.
[RupperSort, Iupper] = sort(Rupper, 1);
[RlowerSort, Ilower] = sort(Rlower, 1);

% compute the overlapping area OA_j among classes of feature X_j
OA = zeros(numFeat,1);
psi = zeros(length(class),length(class), numFeat);
store = [];
for j = 1:numFeat
    for c = 1:length(class)-1
        for k = c+1:length(class)
            % try Rupper and Rlower vs RupperSort and RlowerSort
            psi(c, k, j) = max(RupperSort(c,j) - RlowerSort(k,j), 0);
        end
    end
    OA(j) = sum(sum(psi(:,:,j)));
end

% compute the Area Coefficient of feature X_j
AC = OA ./ (max(Rupper, [], 1) - max(Rlower, [], 1))';

% compute the normalized area coefficient
NAC = AC / max(AC);

% compute the weight of the jth feature of X
w = 1 - NAC;

% get the indices of which features are have the most weight
% the first n values in wInd are the n most relevant features
[~, wInd] = sort(w, 'descend');

%% PCA
% normalize X data
demeanX = X - sum(X,1)/numSamp;

% % calculate covariance matrix
Sigma = 1/numSamp * (X' * X);

% % calculate eigenvectors
%[U, S, V] = svd(Sigma);
% 
% % extract desired number of features
% k = floor(numSamp/100);
% Ureduce = U(:, 1:k);
% 
% % get k principal component vectors
% Z = Ureduce' * X;

%% Dataset
%%%% standardize the data
% take the mean of each feature
mu_feat = mean(X, 1);

% take the standard deviation of each feature
std_feat = std(X, 0, 1);

% standardize X
X_standardized = (X - mu_feat) ./ std_feat;

% EITHER extract k most useful features
% OR extract number of features w/ weight > 0.9
k = 150;
X_useful = X_standardized(:, wInd(1:k));
X_useful = [X_useful ones(size(X_useful, 1), 1)];

% random seed
seed = 42;
rng(seed);

% folds for splitting data
folds = 6;

% initialize train and test data
X_fold = cell(folds, 1);
y_fold = cell(folds, 1);

% split dataset into f folds
for c = 1:length(class)
   % random permutation of indices
   perm = randperm(classIndicesEnd(c) - classIndicesBegin(c) + 1) - 1;
   
   % number of indices/samples per fold
   numInd = floor(length(perm) / folds);
   modulus = mod(length(perm), folds);
   start = 1;
   for f = 1:folds
       if (f > modulus)
           ind = start:start+numInd-1;
           start = start + numInd;
       else
           ind = start:start+numInd;
           start = start + numInd;
       end
       X_fold{f} = [X_fold{f}; X_useful(classIndicesBegin(c)+perm(ind), :)];
       y_fold{f} = [y_fold{f}; y(classIndicesBegin(c)+perm(ind))];
       
   end
end


% % One hot encode y for creating multiple classifiers
% % y == 1 when in the class
% % y == -1 when not in the class
% y_multi = cell(folds, length(class));
% for f = 1:folds
%     for c = 1:length(class)
%        y_multi{f,c} = 2*(y_fold{f} == class(c)) - 1;
%     end
% end
%%%% STANDARDIZE THE DATA?
%%%% DO I USED -1 OR 0 WHEN Y IS NOT IN THE CLASS??
%%%% NOTE: MIGHT USE LAST OR SECOND LAST SLIDE ON LEC4 INSTEAD


% Separate the data into sets of folds for training and testing
% training and testing for each set of folds
train_X_fold = cell(folds, 1);
train_y_fold = cell(folds, 1);

test_X_fold = cell(folds, 1);
test_y_fold = cell(folds, 1);

% for each fold
for f = 1:folds
    
    % extract training data by taking (#fold-1) folds for training
    for i = 1:folds
        % use every fold but the current for training
        if (i ~= f)
            train_X_fold{f} = [train_X_fold{f}; X_fold{i}];
            train_y_fold{f} = [train_y_fold{f}; y_fold{i}];
        end
    end

    % extract testing data from current fold f
    test_X_fold{f} = X_fold{f};
    test_y_fold{f} = y_fold{f};

end


%% Logistic Regression
% regularization constants
lambda = [-3 -2 -1 0 1 2 3];
lambda = 10 .^ lambda;

% stores parametrs that are solved for
theta_Logistic = cell(folds, length(lambda));

% stores training and testing accuracy
trainAccLog = zeros(folds, length(lambda));
testAccLog  = zeros(folds, length(lambda));

for f = 1:folds
    % try each regularization constant
    for reg = 1:length(lambda)
        % minimize loss function to solve for parameters
        cvx_begin
            variables theta_log(k+1, length(class));
            expression Reg(length(class));
            Loss = log(sum(exp(train_X_fold{f} * theta_log),2)) - diag(train_X_fold{f}*theta_log(:, train_y_fold{f}));
            for c = 1:length(class)
                Reg(c) = lambda(reg) * norm(theta_log(:,c));
            end
            minimize(sum(Loss) + sum(Reg));
        cvx_end

        % store parameters solved for
        theta_Logistic{f, reg} = theta_log;

        % prediction on training data
        z = train_X_fold{f} * theta_log;
        sig = 1 ./ (1 + exp(-z));
        [maxProb pred] = max(sig, [] , 2);
        acc = sum(pred == train_y_fold{f})/length(pred)*100;
        trainAccLog(f, reg) = acc;

        % prediction on testing data
        z = test_X_fold{f} * theta_log;
        sig = 1 ./ (1 + exp(-z));
        [maxProb pred] = max(sig, [] , 2);
        acc = sum(pred == test_y_fold{f})/length(pred)*100;
        testAccLog(f, reg) = acc;
    end    
    
end



%% Test Logistic Regression

% set regularization constant 
reg = 5;

% initialize confusion matrix
conf_matrix_Log = cell(folds, 1);

% create confusion matrices for testing each fold
for f = 1:folds    
    z = test_X_fold{f} * theta_Logistic{reg};
    sig = 1 ./ (1 + exp(-z));
    [maxProb pred] = max(sig, [] , 2);
    conf_matrix_Log{f} = zeros(length(class));
    for i = 1:length(class)
        for j = 1:length(class)
            conf_matrix_Log{f}(i, j) = sum(test_y_fold{f} == i & pred == j);
        end
    end
end
    
%% SVM
% train SVM for each class
% loss function, hinge loss
% regularization constants
lambda = [-3 -2 -1 0 1 2 3];
lambda = 10 .^ lambda;

% stores parametrs that are solved for
theta_SVM = cell(folds, length(lambda));

% stores training and testing accuracy
trainAccSVM = zeros(folds, length(lambda));
testAccSVM  = zeros(folds, length(lambda));

% for each set of folds
for f = 1:folds
    % matrix of 1's for adding in loss function
    n = length(train_y_fold{f});
    m = max(train_y_fold{f});
    oneMat = ones(n, m);
    vec = 1:n;
    idx = sub2ind([n, m], vec', train_y_fold{f});
    oneMat(idx) = 0;
    
    % try each regularization constant
    for reg = 1:length(lambda)
        
        % minimize loss function to solve for parameters
        cvx_begin
            % variables
            variables theta_svm(k+1, length(class));
            expression Reg(length(class));
            
            % Loss function
            Loss = (train_X_fold{f} * theta_svm) - diag(train_X_fold{f} * theta_svm(:, train_y_fold{f}))*ones(1,length(class));
            Loss = Loss + oneMat;   
            Loss = sum(max(Loss, [], 2)); 
            
            % Regularization term
            for c = 1:length(class)
                Reg(c) = lambda(reg) * norm(theta_svm(:,c));
            end
            
            % minimize the loss + regularization
            minimize(sum(Loss) + sum(Reg));
        cvx_end

        % store parameters solved for
        theta_SVM{f, reg} = theta_svm;

        % prediction on training data
        z = train_X_fold{f} * theta_svm;
        sig = 1 ./ (1 + exp(-z));
        [maxProb pred] = max(sig, [] , 2);
        acc = sum(pred == train_y_fold{f})/length(pred)*100;
        trainAccSVM(f, reg) = acc;

        % prediction on testing data
        z = test_X_fold{f} * theta_svm;
        sig = 1 ./ (1 + exp(-z));
        [maxProb pred] = max(sig, [] , 2);
        acc = sum(pred == test_y_fold{f})/length(pred)*100;
        testAccSVM(f, reg) = acc;
    end   
end


%% Test SVM
% set regularization constant 
% 5 seemed to be best value since it produced overall
% best generalization for each fold
reg = 5;

% initialize confusion matrix
conf_matrix_SVM = cell(folds, 1);

% create confusion matrices for testing each fold
for f = 1:folds    
    z = test_X_fold{f} * theta_Logistic{reg};
    sig = 1 ./ (1 + exp(-z));
    [maxProb pred] = max(sig, [] , 2);
    conf_matrix_SVM{f} = zeros(length(class));
    for i = 1:length(class)
        for j = 1:length(class)
            conf_matrix_SVM{f}(i, j) = sum(test_y_fold{f} == i & pred == j);
        end
    end
end


save('train_test_folds.mat', 'train_X_fold', 'train_y_fold');
save('LR_Results.mat', 'theta_Logistic', 'trainAccLog', 'testAccLog', 'conf_matrix_Log');
save('SVM_Results.mat', 'theta_SVM', 'trainAccSVM', 'testAccSVM', 'conf_matrix_SVM');



%% Naive Bayes
% stores data as discrete values in folds
train_X_fold_Disc1 = cell(folds, 1);
train_X_fold_Disc2 = cell(folds, 1);
test_X_fold_Disc1  = cell(folds, 1);
test_X_fold_Disc2  = cell(folds, 1);


for f = 1:folds
   train_X_fold_Disc1{f} = zeros(size(train_X_fold{f}(:,1:k))); 
   test_X_fold_Disc1{f}  = zeros(size(test_X_fold{f}(:,1:k))); 
end


%%% Discretization Method 1 %%%
% set gene expression values to discrete values
% 1  - overexpressed
% 0  - normally expressed
% -1 - underexpressed
thres_over  = 1;
thres_under = -1;

% discretize data based on overexpression or underexpression
for f = 1:folds
    train_X_fold_Disc1{f} = train_X_fold_Disc1{f} + (train_X_fold{f}(:,1:k) > thres_over | train_X_fold{f}(:,1:k) < thres_under);
    test_X_fold_Disc1{f} = test_X_fold_Disc1{f} + (test_X_fold{f}(:,1:k) > thres_over | test_X_fold{f}(:,1:k) < thres_under);
end

% initialize variables to store probabilities, predictions, and accuracies
p = cell(folds, 1);
p_c = zeros(folds, length(class));
p_x_given_y = [];
p_y_given_x = [];
pred = cell(folds, 1);
predTrain = cell(folds, 1);
trainAccNBDisc1 = zeros(folds, 1);
testAccNBDisc1 = zeros(folds, 1);
store = [139, 17, 5, 21, 20, 0, 0];

% calculate the probability of a given class pc and p(x_j | y=c)
for f = 1:folds
%%% training
    for c = 1:length(class)
        p_c(f, c) = sum(train_y_fold{f} == class(c)) / length(train_y_fold{f});
        p{f} = [p{f}; sum(train_X_fold_Disc1{f}(train_y_fold{f}==c,:)) / sum(y==c)];
    end
    
%%% testing 
    % training accuracy
    for n = 1:size(train_X_fold{f}, 1)
        p_x_given_y = 1 + train_X_fold_Disc1{f}(n,:) .* p{f} + (1 - train_X_fold_Disc1{f}(n,:)) .* (1 - p{f});
        p_y_given_x = p_c(f, :)' .* prod(p_x_given_y, 2);
        [M I] = max(p_y_given_x);
        predTrain{f} = [predTrain{f}; I];
%         store = [store; p_y_given_x' train_y_fold{f}(n,:) predTrain{f}(n)];
    end
    
    trainAccNBDisc1(f) = sum(predTrain{f} == train_y_fold{f}) / length(predTrain{f}) * 100;

    % testing accuracy
    for n = 1:size(test_X_fold{f}, 1)
        p_x_given_y = 1 + test_X_fold_Disc1{f}(n,:) .* p{f} + (1 - test_X_fold_Disc1{f}(n,:)) .* (1 - p{f});
        p_y_given_x = p_c(f, :)' .* prod(p_x_given_y, 2);
        [M I] = max(p_y_given_x);
        pred{f} = [pred{f}; I];
        store = [store; p_y_given_x' test_y_fold{f}(n,:) pred{f}(n)];
    end
    
    testAccNBDisc1(f) = sum(pred{f} == test_y_fold{f}) / length(pred{f}) * 100;
end


%%
%%% Discretization Method 2 %%%
% EWD - Equal Width Distribution

% initialize training and testing data
for f = 1:folds
   train_X_fold_Disc2{f} = zeros(size(train_X_fold{f}(:,1:k))); 
   test_X_fold_Disc2{f}  = zeros(size(test_X_fold{f}(:,1:k)));
end
   
% Divide the data into k (user-defined) discrete bins
% Width = (max(X) - min(X)) / k
numBins = 10;
bins = (max(max(X_useful)) - min(min(X_useful))) / numBins;

% discretize data based on overexpression or underexpression
for f = 1:folds
    thres_over  = min(min(train_X_fold{f}));
    thres_under = thres_over + bins;
    for b = 1:numBins
        train_X_fold_Disc2{f} = train_X_fold_Disc2{f} + b*(train_X_fold{f}(:,1:k) >= thres_over & train_X_fold{f}(:,1:k) < thres_under);
        test_X_fold_Disc2{f}  = test_X_fold_Disc2{f}  + b*(test_X_fold{f}(:,1:k)  >= thres_over & test_X_fold{f}(:,1:k)  < thres_under);
        thres_over  = min(min(train_X_fold{f})) + b*bins;
        thres_under = thres_over + bins;
    end
end

% initialize variables to store probabilities, predictions, and accuracies
p = cell(folds, 1);
p_c = zeros(folds, length(class));
p_x_given_y = [];
p_y_given_x = [];
pred = cell(folds, 1);
predTrain = cell(folds, 1);
trainAccNBDisc2 = zeros(folds, 1);
testAccNBDisc2 = zeros(folds, 1);
store = [139, 17, 5, 21, 20, 0, 0];

% calculate the probability of a given class pc and p(x_j | y=c)
for f = 1:folds
%%% training
    for c = 1:length(class)
        p_c(f, c) = sum(train_y_fold{f} == class(c)) / length(train_y_fold{f});
        p{f} = [p{f}; sum(train_X_fold_Disc2{f}(train_y_fold{f}==c,:)) / sum(y==c)];
    end
    
%%% testing 
    % training accuracy
    for n = 1:size(train_X_fold{f}, 1)
%         p_x_given_y = prod(p{f} .^ train_X_fold_Disc2{f}(n, :), 2);
        p_x_given_y = sum(train_X_fold_Disc2{f}(n,:) .* log10(p{f}), 2);
        p_y_given_x = p_x_given_y .* p_c(f, :)';
        
%         p_y_given_x(i,:) = prod(p' .^ X_test(i,:),2);
%         p_y_given_x(i,:) = p_y_given_x(i,:) .* py';
        
        [M I] = max(p_y_given_x);
        predTrain{f} = [predTrain{f}; I];
        store = [store; p_y_given_x' train_y_fold{f}(n,:) predTrain{f}(n)];
    end
    
    trainAccNBDisc2(f) = sum(predTrain{f} == train_y_fold{f}) / length(predTrain{f}) * 100;

    % testing accuracy
    for n = 1:size(test_X_fold{f}, 1)
        p_x_given_y = sum(test_X_fold_Disc2{f}(n,:) .* log10(p{f}), 2);
        p_y_given_x = p_x_given_y .* p_c(f, :)';
        [M I] = max(p_y_given_x);
        pred{f} = [pred{f}; I];
        store = [store; p_y_given_x' test_y_fold{f}(n,:) pred{f}(n)];
    end
    
    testAccNBDisc2(f) = sum(pred{f} == test_y_fold{f}) / length(pred{f}) * 100;
end


