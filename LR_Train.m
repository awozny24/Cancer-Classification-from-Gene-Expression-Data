function [theta] = LR_Train(trainX, y, lambda)
% Takes a matrix of samples X, label vector y, and regularization
% parameter lambda, and trains a Logistic Regression model
    % Input: 
        % X: matrix of n samples by m features
        % y: labels for each samples (n-vector)
        % lambda: regularization parameter
    % Output:
        % theta: trained parameters
        
    % get X data
    X = trainX;
    
    % get number of features
    numFeat = size(X,2);

    % minimize loss function to solve for parameters
    cvx_begin
        % define variables
        variables theta_log(numFeat, max(y));
        
        % regularization variable
        expression Reg(max(y));
        
        % calculate the loss
        Loss = log(sum(exp(X * theta_log),2)) - diag(X*theta_log(:, y));
        
        % calculate the regularization value
        for c = 1:max(y)
            Reg(c) = lambda * norm(theta_log(:,c));
        end
        
        % minimize the total loss + regularization
        minimize(sum(Loss) + sum(Reg));
        
    cvx_end

    % return the parameters
    theta = theta_log;
    
end

