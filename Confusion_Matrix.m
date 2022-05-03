function conf_matrix = Confusion_Matrix(X, y, theta, reg, model)

    % initialize confusion matrix
    conf_matrix = zeros(max(y));

    % multiply by parameters
    z = X * theta;
    
    % make prediction for logistic regression
    if (model=="LR" || model=="Logistic Regression" || model == "LogisticRegression" || ...
    model == "Log" || model == "log" || model == "Logistic" || model == "logistic")
        sig = 1 ./ (1 + exp(-z));
        [maxProb pred] = max(sig, [] , 2);
    % make prediction for svm
    elseif (model == "SVM" || model == "svm")
        [maxProb pred] = max(z, [] , 2);
    else
        fprintf("Please Enter Correct Model%s", "!");
    end
    
    % create confusion matrix for correct and incorrect predictions
    for i = 1:max(y)
        for j = 1:max(y)
            conf_matrix(i, j) = sum(y == i & pred == j);
        end
    end
    
    % return confusion matrix
    conf_matrix = conf_matrix(:,:);

end

