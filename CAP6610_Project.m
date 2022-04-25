% lung adenocarcinomas (n = 127)
% Other adenocarcinomas (n = 12) were suspected to be extrapulmonary metastases
% normal lung (n = 17) specimens
% SCLC (n = 6) cases
% squamous cell lung carcinomas (n = 21)
% pulmonary carcinoids (n = 20)

clc; clear all; close all;

%% Load Gene Expression Data
run = false;
if (run)
numvars = 203;
type = 'double';
vartype = repmat({type},1,203);
opts = spreadsheetImportOptions('NumVariables', numvars, 'VariableTypes', vartype);
opts.DataRange = 'C2';
data = readmatrix('dataseta_12600gene.xls', opts);
X = data';

% import gene names cells
clear opts
opts = spreadsheetImportOptions('VariableNames', 'gene');
opts.DataRange = 'B2';
genes = readmatrix('dataseta_12600gene.xls', opts);
end

%% Load Classification Types
% import data into cells
run = false;
if (run)
clear opts
opts = spreadsheetImportOptions('NumVariables', numvars);
opts.DataRange = 'C1:GW1';
labels = readmatrix('dataseta_12600gene.xls', opts);

% labels for different lung tissue samples
cancerName = {'lung adenocarcinomas',
              'normal lung',
              'SCLC',
              'squamous cell lung carcinomas',
              'pulmonary carcinoids'};

% class numbers for corresponding labels
class = {1, 2, 3, 4, 5};

% map the labels to a number
map = containers.Map(class, cancerName);

% store class for each corresponding label in variable y
y = zeros(numvars, 1);
for n = 1:numvars
    % 'lung adenocarcinomas', 
    if (labels{1,n}(1,1:2) == 'AD')
        y(n) = 1;
    % 'normal lung',    
    elseif (labels{1,n}(1,1:2) == 'NL')    
        y(n) = 2;        
    % 'SCLC',    
    elseif (labels{1,n}(1,1:2) == 'SM')    
        y(n) = 3;    
    % 'squamous cell lung carcinomas',    
    elseif (labels{1,n}(1,1:2) == 'SQ')    
        y(n) = 4;
    % 'pulmonary carcinoids',    
    elseif (labels{1,n}(1,1:2) == 'CO')    
        y(n) = 5;
    end
end

class = cell2mat(class);
numSamp = size(X, 1);
numFeat = size(X, 2);
end


%% Load Data
load('dataLoad.mat')


%% ERGS and Divide into Folds
% perform ERGs, Standardization, and add bias
k = 50;
X_useful = ERGS(X, y, k, true, true);

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

% Separate the data into sets of folds for training and testing
% training and testing for each set of folds
train_X_fold = cell(folds, 1);
train_y_fold = cell(folds, 1);

val_data = false;
if (val_data)
    valTest_X_fold = cell(folds, 1);
    valTest_y_fold = cell(folds, 1);
end

test_X_fold = cell(folds, 1);
test_y_fold = cell(folds, 1);

% for each fold
for f = 1:folds
    
    % extract training data by taking (#fold-1) folds for training
    count = 1;
    for i = 1:folds
        % use every fold but the current for training
        if (i ~= f)
           
            if (val_data)
                if (count < 5)            
                    train_X_fold{f} = [train_X_fold{f}; X_fold{i}];
                    train_y_fold{f} = [train_y_fold{f}; y_fold{i}];
                    count = count + 1;
                else
                    valTest_X_fold{f} = X_fold{i};
                    valTest_y_fold{f} = y_fold{i};
                end
            else
                train_X_fold{f} = [train_X_fold{f}; X_fold{i}];
                train_y_fold{f} = [train_y_fold{f}; y_fold{i}];
            end
        end
    end

    % extract testing data from current fold f
    test_X_fold{f} = X_fold{f};
    test_y_fold{f} = y_fold{f};
    
end


%% Logistic Regression
% regularization constants
lambdaLR = [-3 -2 -1 0 1 2 3];
lambdaLR = 10 .^ lambdaLR;

% run training if not yet trained
runLR = false; 
if (runLR)
    
    % stores parametrs that are solved for
    theta_Logistic = cell(folds, length(lambdaLR));
    
    % stores training and testing accuracy
    trainAccLog = zeros(folds, length(lambdaLR));
    testAccLog  = zeros(folds, length(lambdaLR));

    % for each fold
    for f = 1:folds
        % try each regularization constant
        for reg = 1:length(lambdaLR)

            % train the logistic regression model
            theta_log = LR_Train(train_X_fold{f}, train_y_fold{f}, lambdaLR(reg));

            % store parameters solved for
            theta_Logistic{f, reg} = theta_log;

            % prediction on training data
            trainAccLog(f, reg) = LR_Predict(train_X_fold{f}, train_y_fold{f}, theta_log);

            % prediction on testing data
            testAccLog(f, reg) = LR_Predict(test_X_fold{f}, test_y_fold{f}, theta_log);

        end    

    end
    
% if model already trained, load results
else
   load("LR_Results_k" + num2str(k) + ".mat"); 
end




%% Test Logistic Regression
% set regularization constant (5 seemed to be best value since it produced 
% overall best generalization for each fold
reg = find(lambdaLR == 10^0);

% create confusion matrix
confMatLog = zeros(length(class));

% calculate confusion matrix
for f = 1:folds
    confMatLog = confMatLog + Confusion_Matrix(test_X_fold{f}, test_y_fold{f}, theta_Logistic{f, reg}, reg, "Logistic");
end

%% SVM
% train SVM for each class
% loss function, hinge loss
% regularization constants
lambdaSVM = [-3 -2 -1 0 1 2 3];
lambdaSVM = 10 .^ lambdaSVM;

runSVM = false;
if (runSVM)
    % stores parameters that are solved for
    theta_SVM = cell(folds, length(lambdaSVM));
    
    % stores training and testing accuracy
    trainAccSVM = zeros(folds, length(lambdaSVM));
    testAccSVM  = zeros(folds, length(lambdaSVM));

    % for each set of folds
    for f = 1:folds

        % try each regularization constant
        for reg = 1:length(lambdaSVM)

            % train SVM for parameters theta
            theta_svm = SVM_Train(train_X_fold{f}, train_y_fold{f}, lambdaSVM(reg));

            % store trained parameters
            theta_SVM{f, reg} = theta_svm;

            % prediction on training data
            trainAccSVM(f, reg) = SVM_Predict(train_X_fold{f}, train_y_fold{f}, theta_svm);

            % prediction on testing data
            testAccSVM(f, reg) = SVM_Predict(test_X_fold{f}, test_y_fold{f}, theta_svm);
        end   
    end
else
    load("SVM_Results_k" + num2str(k) + ".mat");
end


%% Test SVM
% set regularization constant (5 seemed to be best value since it produced 
% overall best generalization for each fold
reg = find(lambdaSVM == 10^1);

% create confusion matrix
confMatSVM = zeros(length(class));

% create confusion matrices for testing each fold
for f = 1:folds    
    confMatSVM = confMatSVM + Confusion_Matrix(test_X_fold{f}, test_y_fold{f}, theta_SVM{f, reg}, reg, "SVM"); 
end



% %% Save Parameters and Results if needed
% save('train_test_folds.mat', 'train_X_fold', 'train_y_fold');
% save('LR_Results.mat', 'theta_Logistic', 'trainAccLog', 'testAccLog', 'conf_matrix_Log');
% save('SVM_Results.mat', 'theta_SVM', 'trainAccSVM', 'testAccSVM', 'confMatSVM');

%% SVM Gaussian Kernel
% regularization parameters
lambdaRBF = [-3 -2 -1 0 1 2 3];
lambdaRBF = 10 .^ lambdaRBF;

% stores parameters that are solved for
theta_SVM_RBF = cell(folds, length(lambdaRBF));

% stores training and testing accuracy
trainAccSVMRBF = zeros(folds, length(lambdaRBF));
testAccSVMRBF  = zeros(folds, length(lambdaRBF));

run = false;
if (run)
% for each set of folds
for f = 1:folds

    % try each regularization constant
    for reg = 1:length(lambdaRBF)

        % get transformed features for kernel trick
        trainSz = size(train_X_fold{f},1);
        testSz = size(test_X_fold{f},1);
        dataF = [train_X_fold{f}; test_X_fold{f}];
        dataF = SimilarityRBF(dataF, true, false);
        
        % train SVM for parameters theta
        theta_svm_rbf = SVM_Train(dataF(1:trainSz,:), train_y_fold{f}, lambdaRBF(reg));

        % store trained parameters
        theta_SVM_RBF{f, reg} = theta_svm_rbf;

        % prediction on training data
        trainAccSVMRBF(f, reg) = SVM_Predict(dataF(1:trainSz,:), train_y_fold{f}, theta_svm_rbf);

        % prediction on testing data
        testAccSVMRBF(f, reg) = SVM_Predict(dataF(trainSz+1:end,:), test_y_fold{f}, theta_svm_rbf);
    end   
end
else
   load("SVM_RBF_Results_k" + num2str(k) + ".mat"); 
end



%% Test SVM Gaussian Kernel
% set regularization constant (5 seemed to be best value since it produced 
% overall best generalization for each fold
reg = find(lambdaRBF == 10^0);

% create confusion matrix
confMatRBF = zeros(length(class));

% create confusion matrices for testing each fold
for f = 1:folds  
    % get transformed features for kernel trick
    trainSz = size(train_X_fold{f},1);
    testSz = size(test_X_fold{f},1);
    dataF = [train_X_fold{f}; test_X_fold{f}];
    dataF = SimilarityRBF(dataF, true, false);
    
    % calculate confusion matrix
    confMatRBF = confMatRBF + Confusion_Matrix(dataF(trainSz+1:end,:), test_y_fold{f}, theta_SVM_RBF{f, reg}, reg, "SVM"); 
end


%% SVM Polynomial Kernel
% regularization parameters
lambdaPoly = [-9, -8, -7];
lambdaPoly = 10 .^ lambdaPoly;

% stores parameters that are solved for
theta_SVM_Poly = cell(folds, length(lambdaPoly));

% stores training and testing accuracy
trainAccSVMPoly = zeros(folds, length(lambdaPoly));
testAccSVMPoly  = zeros(folds, length(lambdaPoly));


run = false;
if (run)
% for each set of folds
for f = 1:folds
    
    % try each regularization constant
    for reg = 1:length(lambdaPoly)
        
        % combine data for kernel trick transformation
        trainSz = size(train_X_fold{f},1);
        dataF = [train_X_fold{f}; test_X_fold{f}];
        
        % parameters for transformation
        slope = 1;
        intercept = 40; % best values were in order: 40, -111, -35
                        % 100 generalized well too
        degree = 1;     % best degree was 1, then 2
        bias_exist = true;
        add_bias = false;
        
        % use polynomial kernel trick on data
        dataF = SimilarityPoly(dataF, degree, slope, intercept, bias_exist, add_bias);
        
        % train SVM for parameters theta
        theta_svm_poly = SVM_Train(dataF(1:trainSz,:), train_y_fold{f}, lambdaPoly(reg));

        % store trained parameters
        theta_SVM_Poly{f, reg} = theta_svm_poly;

        % prediction on training data
        trainAccSVMPoly(f, reg) = SVM_Predict(dataF(1:trainSz,:), train_y_fold{f}, theta_svm_poly);

        % prediction on testing data
        testAccSVMPoly(f, reg) = SVM_Predict(dataF(trainSz+1:end,:), test_y_fold{f}, theta_svm_poly);
    end   
end
else
   load("SVM_Poly_Results_k" + num2str(k) + ".mat"); 
end


%% Test SVM Polynomial Kernel
% set regularization constant (4 seemed to be best value since it produced 
% overall best generalization for each fold
reg = find(lambdaPoly == 10^(-8));

% create confusion matrix
confMatPoly = zeros(length(class));

% create confusion matrices for testing each fold
for f = 1:folds  
    % combine data for kernel trick transformation
    trainSz = size(train_X_fold{f},1);
    testSz = size(test_X_fold{f},1);
    dataF = [train_X_fold{f}; test_X_fold{f}];
    
    % parameters for transformation
    slope = 1;
    intercept = 40; % best values were in order: 40, -111, -35
                    % 100 generalized well too
    degree = 1;     % best degree was 1, then 2
    bias_exist = true;
    add_bias = false;
        

    % use polynomial kernel trick on data
    dataF = SimilarityPoly(dataF, degree, slope, intercept, bias_exist, add_bias);

    % calculate confusion matrix
    confMatPoly = confMatPoly + Confusion_Matrix(dataF(trainSz+1:end,:), test_y_fold{f}, theta_SVM_Poly{f, reg}, reg, "SVM"); 
end

 
%% Naive Bayes
% stores data as discrete values in folds
train_X_fold_Disc1 = cell(folds, 1);
train_X_fold_Disc2 = cell(folds, 1);
test_X_fold_Disc1  = cell(folds, 1);
test_X_fold_Disc2  = cell(folds, 1);

% initialize train and test folds
for f = 1:folds
   train_X_fold_Disc1{f} = zeros(size(train_X_fold{f}(:,1:k))); 
   test_X_fold_Disc1{f}  = zeros(size(test_X_fold{f}(:,1:k))); 
end


%% Discretization Method
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
predProb = cell(folds, 1);
predProbTrain = cell(folds, 1);
NBProb = cell(folds,1);
NBProbTrain = cell(folds,1);

% calculate the probability of a given class pc and p(x_j | y=c)
for f = 1:folds
%%%%%%%%%%% training %%%%%%%%%%%
    for c = 1:length(class)
        p_c(f, c) = sum(train_y_fold{f} == class(c)) / length(train_y_fold{f});
        p{f} = [p{f}; sum(train_X_fold_Disc1{f}(train_y_fold{f}==c,:)) / sum(train_y_fold{f}==c)];
    end
    
%%%%%%%%%%% testing  %%%%%%%%%%%
    % training accuracy
    for n = 1:size(train_X_fold{f}, 1)
        p_y_given_x = log(p_c(f,:)') + sum(train_X_fold_Disc1{f}(n,:) .* log(1+p{f}), 2) + sum((1 - train_X_fold_Disc1{f}(n,:)) .* log(1 - p{f} + 1),2);
        NBProbTrain{f} = [NBProbTrain{f}; p_y_given_x'];
        [M I] = max(p_y_given_x);
        predProbTrain{f} = [predProbTrain{f}; M];
        predTrain{f} = [predTrain{f}; I];
    end
    
    % calculate training accuracy
    trainAccNBDisc1(f) = sum(predTrain{f} == train_y_fold{f}) / length(predTrain{f}) * 100;

    % testing accuracy
    for n = 1:size(test_X_fold{f}, 1)
        p_y_given_x = log(p_c(f,:)') + sum(test_X_fold_Disc1{f}(n,:) .* log(1+p{f}), 2) + sum((1 - test_X_fold_Disc1{f}(n,:)) .* log(1 - p{f} + 1),2);
        NBProb{f} = [NBProb{f}; p_y_given_x'];
        [M I] = max(p_y_given_x);
        predProb{f} = [predProb{f}; M];
        pred{f} = [pred{f}; I];
    end
    
    % calculate testing accuracy
    testAccNBDisc1(f) = sum(pred{f} == test_y_fold{f}) / length(pred{f}) * 100;
end

%% Test Naive Bayes

confMatNB = zeros(max(y), max(y));
% create confusion matrix for correct and incorrect predictions
for i = 1:max(y)
    for j = 1:max(y)
        for f = 1:folds
            confMatNB(i, j) = confMatNB(i,j) + sum(test_y_fold{f} == i & pred{f} == j);
        end
    end
end
    

    

%% Ensemble Meta-Model w/ LR Base w/o Kernel

%%%% best Logistic Regression Model %%%%
% get best regularization parmater
[~, best] = max(mean(testAccLog), [], 2);

% get average of theta values from each fold
thetaLR = theta_Logistic{1, best};
for f = 2:folds
    thetaLR = thetaLR + theta_Logistic{f, best};
end
thetaLR = thetaLR / folds;

%%%% best SVM Model
% get best regularization parameter
[~, best] = max(mean(testAccSVM), [], 2);
% get average of theta values from each fold
thetaSVM = theta_SVM{1, best};
for f = 2:folds
   thetaSVM = thetaSVM + theta_SVM{f, best};
end
thetaSVM = thetaSVM / folds;


%%%% best SVM Gaussian Kernel Model %%%%
% get best regularization parameter
[~, best] = max(mean(testAccSVMRBF), [], 2);

thetaRBF = theta_SVM_RBF{1, best};
for f = 2:folds
   thetaRBF = thetaRBF + theta_SVM_RBF{f, best};
end
thetaRBF = thetaRBF / folds;

%%%% best SVM Polynomial Kernel w/ Degree 2 Model %%%%
% get best regularization parameter
[~, best] = max(mean(testAccSVMPoly), [], 2);

thetaPoly = theta_SVM_Poly{1, best};
for f = 2:folds
   thetaPoly = thetaPoly + theta_SVM_Poly{f, best}; 
end
thetaPoly = thetaPoly / folds;

% create ensemble stacking training data for each fold
trainX = cell(folds, 1);
testX = cell(folds, 1);

% make prediction based on average theta values
for f = 1:folds
    % Logistic Regression prediction
    trainX{f} = [trainX{f} train_X_fold{f}*thetaLR];
    testX{f} = [testX{f} test_X_fold{f}*thetaLR];

    % SVM Prediction
    trainX{f} = [trainX{f} train_X_fold{f}*thetaSVM];
    testX{f} = [testX{f} test_X_fold{f}*thetaSVM];
      
     % Naive Bayes Prediction
     trainX{f} = [trainX{f} NBProbTrain{f}];
     testX{f} = [testX{f} NBProb{f}];
end


%%% Train Logistic Regression Meta Model w/ No Kernel
% for each fold

% regularization parameters
lambda = [-3 -2 -1 0 1 2 3];
lambda = 10 .^ lambda;

% don't need to train model if trained model has been saved
runMeta = false;
if (runMeta)
    
% stores training and testing accuracy
trainAccMeta = zeros(folds, length(lambda));
testAccMeta  = zeros(folds, length(lambda));

% stores parameters that are solved for
theta_Meta = cell(folds, length(lambda));

for f = 1:folds
    % try each regularization constant
    for reg = 1:length(lambda)

        fprintf("Fold %i Reg %i\n", f, reg);
        % train the logistic regression model
        theta_meta = LR_Train(trainX{f}, train_y_fold{f}, lambda(reg));

        % store parameters solved for
        theta_Meta{f, reg} = theta_meta;

        % prediction on training data
        trainAccMeta(f, reg) = LR_Predict(trainX{f}, train_y_fold{f}, theta_meta);

        % prediction on testing data
        testAccMeta(f, reg) = LR_Predict(testX{f}, test_y_fold{f}, theta_meta);
    end    

end
else
    load("MetaModelCorr_NoKernel_k" + num2str(k) + ".mat");
end

%% Test Ensemble with No Kernel
% set regularization constant (5 seemed to be best value since it produced 
% overall best generalization for each fold
reg = find(lambdaLR == 10^-2);

% create confusion matrix
confMatMetaNoKern = zeros(length(class));

% calculate confusion matrix
for f = 1:folds
    confMatMetaNoKern = confMatMetaNoKern + Confusion_Matrix(testX{f}, test_y_fold{f}, theta_Meta{f, reg}, reg, "Logistic");
end


%% Ensemble Meta-Model w/ LR Base w/o Kernel

% create ensemble stacking training data for each fold
trainX = cell(folds, 1);
testX = cell(folds, 1);

% make prediction based on average theta values
for f = 1:folds
    % Logistic Regression prediction
    trainX{f} = [trainX{f} train_X_fold{f}*thetaLR];
    testX{f} = [testX{f} test_X_fold{f}*thetaLR];

    % SVM Prediction
    trainX{f} = [trainX{f} train_X_fold{f}*thetaSVM];
    testX{f} = [testX{f} test_X_fold{f}*thetaSVM];

    % SVM Gaussian Kernel Prediction
     % get transformed features for kernel trick
     trainSz = size(train_X_fold{f},1);
     testSz = size(test_X_fold{f},1);
     dataF = [train_X_fold{f}; test_X_fold{f}];
     dataF = SimilarityRBF(dataF, true, false);
     
     % prediction
     trainX{f} = [trainX{f} dataF(1:trainSz,:)*thetaRBF];
     testX{f} = [testX{f} dataF(trainSz+1:end,:)*thetaRBF];
     
    % SVM Polynomial Degree 1 Kernel Prediction
     % parameters for transformation
     slope = 1;
     intercept = 40;
     degree = 1;
     bias_exist = true;
     add_bias = false;
     
     % combine data for kernel trick transformation
     dataF = [train_X_fold{f}; test_X_fold{f}];
     dataF = SimilarityPoly(dataF, degree, slope, intercept, bias_exist, add_bias);
     
     % prediction
     trainX{f} = [trainX{f} dataF(1:trainSz,:)*thetaPoly];
     testX{f} = [testX{f} dataF(trainSz+1:end,:)*thetaPoly];
      
     % Naive Bayes Prediction
     trainX{f} = [trainX{f} NBProbTrain{f}];
     testX{f} = [testX{f} NBProb{f}];
end

%%% Train Logistic Regression Meta Model

% regularization parameters
lambda = [-3 -2 -1 0 1 2 3];
lambda = 10 .^ lambda;

% don't need to run if trained model is saved
runMeta = false;
if (runMeta)
    
% stores training and testing accuracy
trainAccMeta = zeros(folds, length(lambda));
testAccMeta  = zeros(folds, length(lambda));

% stores parameters that are solved for
theta_Meta = cell(folds, length(lambda));

for f = 1:folds
    % try each regularization constant
    for reg = 1:length(lambda)

        fprintf("Fold %i Reg %i\n", f, reg);
        % train the logistic regression model
        theta_meta = LR_Train(trainX{f}, train_y_fold{f}, lambda(reg));

        % store parameters solved for
        theta_Meta{f, reg} = theta_meta;

        % prediction on training data
        trainAccMeta(f, reg) = LR_Predict(trainX{f}, train_y_fold{f}, theta_meta);

        % prediction on testing data
        testAccMeta(f, reg) = LR_Predict(testX{f}, test_y_fold{f}, theta_meta);
    end    

end
else
    load("MetaModelCorr_withKernel_k" + num2str(k) + ".mat");
end

%% Test Ensemble with Kernel
% set regularization constant (5 seemed to be best value since it produced 
% overall best generalization for each fold
reg = find(lambdaLR == 10^(-2));

% create confusion matrix
confMatMetaWithKern = zeros(max(y), max(y));

% calculate confusion matrix
for f = 1:folds
    confMatMetaWithKern = confMatMetaWithKern + Confusion_Matrix(testX{f}, test_y_fold{f}, theta_Meta{f, 1}, 1, "Logistic");
end


%% Function: ERGS.m
%
% <include>ERGS.m</include>

%% Function: LR_Train.m
%
% <include>LR_Train.m</include>

%% Function: SVM_Train.m
%
% <include>SVM_Train.m</include>

%% Function: SimilarityPoly.m
%
% <include>SimilarityPoly.m</include>

%% Function: SimilarityRBF.m
%
% <include>SimilarityRBF.m</include>

%% Function: LR_Predict.m
%
% <include>LR_Predict.m</include>

%% Function: SVM_Predict.m
%
% <include>SVM_Predict.m</include>

%% Function: Confusion_Matrix.m
%
% <include>Confusion_Matrix.m</include>



%% Create Bar Graph
barData = zeros(7, 3);
% plot (best regularization) test accuracy for each model

% number of features
s = ["50", "150", "500"];

% different models
model = ["LR_Results";
         "SVM_Results";
         "NB_Results";
         "SVM_Poly_Results";
         "SVM_RBF_Results";
         "MetaModelCorr_NoKernel";
         "MetaModelCorr_withKernel"];
          
varName = ["Log";
           "SVM";
           "NBDisc1";
           "SVMPoly";
           "SVMRBF";
           "Meta";
           "Meta"];          

% for each model, for each #features, get the best testing accuracy
for m = 1:length(model)
    for a = 1:3
        name = model(m) + "_k" + s(a) + ".mat"; 
        load(name);
        var = eval("testAcc" + varName(m));
        [~, best] = max(mean(var), [], 2);
        barData(m,a) = mean(var(:, best));
    end
end


% plot bar chart
labNames = ["LR";
           "SVM";
           "NBC";
           "SVMPoly";
           "SVMRBF";
           "Meta w/ Kern.";
           "Meta no Kern"];

lab = categorical(labNames);
lab = reordercats(lab,labNames);
bar(lab, barData, 1.0)
yaxis([90,100])
leg = legend("50", "150", "500", "Location", "northwest");
title(leg, "# Features")
title('Test Accuracies of Each Model')
ylabel('% Accuracy')




%% Plot Results
% Logistic Accuracy w/ Regularization
lambda = [-3 -2 -1 0 1 2 3];
folds = 6;
count = 1;
for s = ["50", "150", "500"]
    name = sprintf('LR_Results_k%s.mat', s); 
    load(name)
    plotTrainAcc = mean(trainAccLog);
    plotTestAcc = mean(testAccLog);
    figure(count)
    plot(lambda, plotTrainAcc, lambda, plotTestAcc)
    xlabel('log(\lambda)')
    ylabel('% Accuracy')
    title("Logistic Regression: Regularization Effect on Accuracy w/ " + s + " Features");
    legend('Train Acc.', 'Test Acc.')
    count = count + 1;
end

% SVM Accuracy w/ Regularization
lambda = [-3 -2 -1 0 1 2 3];
folds = 6;
for s = ["50", "150", "500"]
    name = sprintf('SVM_Results_k%s.mat', s); 
    load(name)
    plotTrainAcc = mean(trainAccSVM);
    plotTestAcc = mean(testAccSVM);
    figure(count)
    plot(lambda, plotTrainAcc, lambda, plotTestAcc)
    xlabel('log(\lambda)')
    ylabel('% Accuracy')
    title("SVM: Regularization Effect on Accuracy w/ " + s + " Features");
    legend('Train Acc.', 'Test Acc.')
    count = count + 1;
end

% Naive Bayes Accuracy w/ Regularization
figure(count); count = count + 1;
plot([1:6], trainAccNBDisc1, [1:6], testAccNBDisc1);

% SVM w/ Gaussian Kernel Accuracy w/ Regularization
lambda = [-3 -2 -1 0 1 2 3];
folds = 6;
for s = ["50", "150", "500"]
    name = sprintf('SVM_RBF_Results_k%s.mat', s); 
    load(name)
    plotTrainAcc = mean(trainAccSVMRBF);
    plotTestAcc = mean(testAccSVMRBF);
    figure(count)
    plot(lambda, plotTrainAcc, lambda, plotTestAcc)
    xlabel('log(\lambda)')
    ylabel('% Accuracy')
    title("SVM w/ Gaussian Kernel: Regularization Effect on Accuracy w/ " + s + " Features");
    legend('Train Acc.', 'Test Acc.')
    count = count + 1;
end

% SVM w/ Polynomial Kernel Accuracy w/ Regularization
lambda = [-9 -8 -7];
folds = 6;
for s = ["50", "150", "500"]
    name = sprintf('SVM_Poly_Results_k%s.mat', s); 
    load(name)
    plotTrainAcc = mean(trainAccSVMPoly);
    plotTestAcc = mean(testAccSVMPoly);
    figure(count)
    plot(lambda, plotTrainAcc, lambda, plotTestAcc)
    xlabel('log(\lambda)')
    ylabel('% Accuracy')
    title("SVM w/ Polynomial Kernel (deg 2): Regularization Effect on Accuracy w/ " + s + " Features");
    legend('Train Acc.', 'Test Acc.')
    count = count + 1;
end


% Meta Model w/p Kernels Accuracy w/ Regularization
lambda = [-3 -2 -1 0 1 2 3];
folds = 6;
for s = ["50", "150", "500"]
    name = sprintf('MetaModelCorr_NoKernel_k%s.mat', s); 
    load(name)
    plotTrainAcc = mean(trainAccMeta);
    plotTestAcc = mean(testAccMeta);
    figure(count)
    plot(lambda, plotTrainAcc, lambda, plotTestAcc)
    xlabel('log(\lambda)')
    ylabel('% Accuracy')
    title("Meta Model w/ Kernels: Regularization Effect on Accuracy w/ " + s + " Features");
    legend('Train Acc.', 'Test Acc.')
    count = count + 1;
end

% Meta Model w/p Kernels Accuracy w/ Regularization
lambda = [-3 -2 -1 0 1 2 3];
folds = 6;
for s = ["50", "150", "500"]
    name = sprintf('MetaModelCorr_withKernel_k%s.mat', s); 
    load(name)
    plotTrainAcc = mean(trainAccMeta);
    plotTestAcc = mean(testAccMeta);
    figure(count)
    plot(lambda, plotTrainAcc, lambda, plotTestAcc)
    xlabel('log(\lambda)')
    ylabel('% Accuracy')
    title("Meta Model w/ No Kernels: Regularization Effect on Accuracy w/ " + s + " Features");
    legend('Train Acc.', 'Test Acc.')
    count = count + 1;
end



