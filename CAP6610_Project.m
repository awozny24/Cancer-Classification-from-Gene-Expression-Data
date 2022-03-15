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
% class = {1, 0, 2, 3, 4};
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
%         y(n) = 0;        
%     % 'SCLC',    
%     elseif (labels{1,n}(1,1:2) == 'SM')    
%         y(n) = 2;    
%     % 'squamous cell lung carcinomas',    
%     elseif (labels{1,n}(1,1:2) == 'SQ')    
%         y(n) = 3;
%     % 'pulmonary carcinoids',    
%     elseif (labels{1,n}(1,1:2) == 'CO')    
%         y(n) = 4;
%     end
% end


%% Load Data
load('dataLoad.mat')
class = cell2mat(class);
numSamp = size(X, 1);
numFeat = size(X, 2);

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

%%
% sort the effective ranges of classes in ascending order to compute 
% Overlapping Area (OAi) for each feature Xi.
% NOT SURE IF THIS IS SORTED CORRECTLY FIX
[RupperSort, Iupper] = sort(Rupper, 1);
[RlowerSort, Ilower] = sort(Rlower, 1);

% compute the overlapping area OA_j among classes of feature X_j
%%%% CHECK THIS STEP %%%%%% FIX
%%% HAS TO DO W/ RUPPER AND RUPPERSORT e.g.
OA = zeros(numFeat,1);
psi = zeros(length(class),length(class), numFeat);
store = [];
for j = 1:numFeat
    for c = 1:length(class)-1
        for k = c+1:length(class)
            psi(c, k, j) = max(RupperSort(c,j) - RlowerSort(k,j), 0);
        end
    end
    OA(j) = sum(sum(psi(:,:,j)));
end

% compute the Area Coefficient of feature X_j
%%%%% IF THE ABOVE IS FIXED, FIX THIS %%%%%
%%% HAS TO DO W/ RUPPER AND RUPPERSORT e.g.
AC = OA ./ (max(Rupper, [], 1) - max(Rlower, [], 1))';

% compute the normalized area coefficient
NAC = AC / max(AC);

% compute the weight of the jth feature of X
w = 1 - NAC;

% get the indices of which features are have the most weight
% the first n values in wInd are the n most relevant features
[~, wInd] = sort(w, 'descend');

%% PCA
% % normalize X data
% demeanX = X - sum(X,1)/numSamp;
% 
% % calculate covariance matrix
% Sigma = 1/numSamp * (X' * X);
% 
% % calculate eigenvectors
% [U, S, V] = svd(Sigma);
% 
% % extract desired number of features
% k = floor(numSamp/100);
% Ureduce = U(:, 1:k);
% 
% % get k principal component vectors
% Z = Ureduce' * X;

%% Dataset
% EITHER extract k most useful features
% OR extract number of features w/ weight > 0.9
k = 150;
X_useful = X(:, wInd(1:k));

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
   
   % add 6 different folds
   numInd = length(perm) / folds - 1;
   ind = 1;
   for f = 1:folds
       i
       X_fold{f} = X(X_useful(classIndicesBegin(c)+perm), :);
       y_fold{f} = y(classIndicesBegin(c)+perm);
   end
end

%% SVM
% loss function, hing loss


%% Logistic Regression



%% Naive Bayes






