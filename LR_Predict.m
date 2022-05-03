function [acc, pred] = LR_Predict(X, y, theta)
% Predicts the class of each sample in a matrix of samples, given
% the trained parameters from a Logistic Regression model to do so.
    % Input: 
        % X: matrix of n samples by m features
        % y: labels for each samples (n-vector)
        % theta: parameters for prediction (m parameters by c classes)
    % Output:
        % acc: percent correct
        % pred: vector of predicted classes


    % calculate z for sigmoid
    z = X * theta;
    
    % calculate sigmoid
    sig = 1 ./ (1 + exp(-z));
    
    % find class with the maximum probability
    [maxProb pred] = max(sig, [] , 2);
    
    % calculate the accuracy of the predictions
    acc = sum(pred == y)/length(pred)*100;

end

