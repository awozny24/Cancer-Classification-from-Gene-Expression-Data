function [f] = SimilarityRBF(X, bias_exist, add_bias)
% Takes a matrix of samples X and uses Gaussian kernel to 
% get feature data in a higher dimension
    % Input: 
        % X: matrix of n samples by m features
        % bias_exist: if a bias of 1's exists in the data, get rid of it
        % add_bias: add a bias of 1's after data is converted
    % Output:
        % f: converted X matrix   


    % use f to map X to different feature space
    % number of selected features
    m = size(X, 2);

    % eliminate bias
    if (bias_exist)
        X = X(:, 1:(m-1));
    end

    % initialize f for transformation
    f = zeros(size(X,1), size(X, 1));

    % set l(i) to x(i)
    for i = 1:size(X,1)
        for j = 1:size(X,1)
        % set sigma squared value
        sigma_sq = 100;

        % set x(i)
        x = X(i,:)';

        % set l(i) = x(j)
        l = X(j,:)';

        % use Gaussian kernel
        f(i, j) = exp(-(x - l)'*(x-l)/(2*sigma_sq));
        end
    end       
    
    
    % add bias if requested
    if (add_bias)
        % set last row to ones
        f = [f, ones(size(f,1),1)];
    end

    % return f
    f = f(:,:);
end

