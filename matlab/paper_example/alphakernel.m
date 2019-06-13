function [axy,gxy] = alphakernel(data, varargin)
% [axy,gxy] = alphakernel(data, varargin)  Build a Gaussian/Alpha Kernel
%
% Author: Jay Stanley
% Created: January 2018
%
%   OUTPUTS:
%       axy = affinity kernel (diag = 2, symmetric via k+k') 
%       gxy = graph kernel (eg weight matrix, diag = 0, symmetric via (k+k')/2)
%
%   INPUTS:
%       data = data matrix. Each row is a single datapoint and each column is a feature.
%           
%       varargin:
%           'a'    (default = '2')
%               The alpha parameter in the exponent of the kernel function. Determines the kernel decay rate
%           'eps'  (default = 'knn'; 'knn' or function handle)
%               Kernel bandwidth function. must be'knn' = adaptive or function handle of form eps(dists) e.g .
%                   eps = @(dists) max(min(dists+eye(size(dists))*10^15));
%           'k'    (default = 5; numeric)
%               number of neighbors for knn eps
%           'npca' (default = 100; numeric)
%               number of pca components to use.  0 = no pca.
%
%       REQUIREMENTS: svdpca
%

    p = inputParser;
    
    checkeps = @(x) isa(x, 'function_handle') || strcmp(x,'knn');
    addRequired(p,'data',@isnumeric);
    addParameter(p,'a',2,@isnumeric);
    addParameter(p,'eps', 'knn', checkeps);
    addParameter(p, 'k', 5, @isint);
    addParameter(p, 'npca', 100, @isnumeric);
    addParameter(p, 'distfun', 'euclidean');

    varargin(1:2:end) = lower(varargin(1:2:end)); % force args to lower case
    parse(p,data,varargin{:});
    
    data = p.Results.data;
    a = p.Results.a;
    epsfunc = p.Results.eps;
    k = p.Results.k;
    npca = p.Results.npca;
    distfun = p.Results.distfun;
    if npca == 0
        M = data;
    else
        M = svdpca(data, npca, 'random');
    end
    disp("building kernel..");
    PDX = squareform(pdist(M,distfun));
    
    if strcmp(epsfunc, 'knn')
        knnDST = sort(PDX,1);
        eps = knnDST(k+1,:)';
    else
        eps = epsfunc(PDX);
    end
    
    PDX = bsxfun(@rdivide, PDX, eps);
    kxy = exp(-PDX.^a);
    axy = kxy+kxy';
    %kxy = kxy - diag(diag(kxy));
    gxy = axy/2;
    gxy = gxy - diag(diag(gxy));
    axy = axy/2;
end
    
function answer = isint(n)

    if size(n) == [1 1]
        answer = isreal(n) && isnumeric(n) && round(n) == n &&  n >0;
    else
        answer = false;
    end
end