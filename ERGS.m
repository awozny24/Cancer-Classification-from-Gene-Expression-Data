function [X_useful] = ERGS(X, y, use, standardize, add_bias)
% Takes a matrix of samples X and label vector y and finds the 
% "use" most relevant features if "use" is an integer; otherwise, 
% finds the features that have a greater calculated weight in (0,1)
% than "use"
    % Input: 
        % X: matrix of n samples by m features
        % y: labels for each samples (n-vector)
        % standardize: standardize data if true
        % add_bias: adds bias feature to new data matrix if true
    % Output:
        % X_useful: matrix of features to use for each sample

    % number of samples
    numSamp = size(X,1);
    numFeat = size(X,2);

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
    mu_j_c = zeros(max(y),numFeat);
    p_c = zeros(max(y), 1);
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
    for c = 1:max(y)
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
    psi = zeros(max(y),max(y), numFeat);
    for j = 1:numFeat
        for c = 1:max(y)-1
            for k = c+1:max(y)
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

    if (standardize)
       %%%% standardize the data
        % take the mean of each feature
        mu_feat = mean(X, 1);

        % take the standard deviation of each feature
        std_feat = std(X, 0, 1);

        % standardize X
        X_standardized = (X - mu_feat) ./ std_feat; 
    else
        X_standardized = X;
    end
    

    % if a range is not given, but a fixed number, i.e. k
    if (use > 1)
        % get the indices of which features are have the most weight
        % the first n values in wInd are the n most relevant features
        [~, wInd] = sort(w, 'descend');
        
        % extract k most useful features
        X_useful = X_standardized(:, wInd(1:use));
        
    % if a range is given, extract number of features w/ weight > 0.9    
    else
        % extract features withindices of weights > range k
        X_useful = X_standardized(:, w > use);
    end
    
    % add bias feature if requested
    if (add_bias)
        X_useful = [X_useful ones(size(X_useful, 1), 1)];
    end
    
    % return useful features
    X_useful = X_useful(:,:);
end

