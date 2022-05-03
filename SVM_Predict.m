function [acc, pred] = SVM_Predict(X, y, theta)
% Predicts the class of each sample in a matrix of samples, given
% the trained parameters from an SVM model to do so.
    % Input: 
        % X: matrix of n samples by m features
        % y: labels for each samples (n-vector)
        % theta: parameters for prediction (m parameters by c classes)
    % Output:
        % acc: percent correct
        % pred: vector of predicted classes
    % multiply data times parameters
    z = X * theta;
    
    % make prediction
    [maxProb pred] = max(z, [] , 2);
    
    % determine accuracy
    acc = sum(pred == y)/length(pred)*100;

end

