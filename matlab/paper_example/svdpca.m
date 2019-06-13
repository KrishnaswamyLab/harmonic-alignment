function [Y,U,S,V] = svdpca(X, k, method)

if ~exist('method','var')
    method = 'svd';
end

X = bsxfun(@minus, X, mean(X));

switch method
    case 'svd'
        disp 'PCA using SVD'
        [U,S,V] = svds(X', k);
        Y = X * U;
    case 'random'
        disp 'PCA using random SVD'
        [U,S,V] = randPCA(X', k);
        Y = X * U;
    case 'none'
        disp 'No PCA performed'
        Y = X;
end