function [theta] = SVM_Train(X, y, lambda)
% Takes a matrix of samples X, label vector y, and regularization
% parameter lambda, and trains an SVM model
    % Input: 
        % X: matrix of n samples by m features
        % y: labels for each samples (n-vector)
        % lambda: regularization parameter
    % Output:
        % theta: trained parameters

    % number of features
    numFeat = size(X,2);

    % matrix of 1's for adding in loss function
    n = length(y);
    m = max(y);
    oneMat = ones(n, m);
    vec = 1:n;
    idx = sub2ind([n, m], vec', y);
    oneMat(idx) = 0;
    
    % number of selected features
    numFeat = size(X,2);
    
    % minimize loss function to solve for parameters
    cvx_begin
        % variables
        variables theta_svm(numFeat, max(y));
        expression Reg(max(y));

        % Loss function
        Loss = (X * theta_svm) - diag(X * theta_svm(:, y))*ones(1,max(y));
        Loss = Loss + oneMat;   
        Loss = sum(max(Loss, [], 2)); 

        % Regularization term
        for c = 1:max(y)
            Reg(c) = lambda * norm(theta_svm(:,c));
        end

        % minimize the loss + regularization
        minimize(sum(Loss) + sum(Reg));
    cvx_end
    
    % return trained parameters
    theta = theta_svm;

end

