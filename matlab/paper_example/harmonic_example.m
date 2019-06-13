%%we use this script in the paper to make the corruption experiment.

close all;
clear all;
plt = true;
labels = loadMNISTLabels('train-labels.idx1-ubyte');
imgs = loadMNISTImages('train-images.idx3-ubyte');

N=1000;
NPs = 1; %how many percentages to run
NWs = 1; %number of wavelets
Ns = 1; %number of iterations to run
Ps = linspace(0,1,NPs);
%scale of wavelets  (eg Nf) to use
Ws = [2 8 16 64];
imgs = imgs';
I = eye(784);
% kernel params
k1 = 20;
a1 = 20;
pca1 = 100;
k2 = k1;
a2 = a1;
pca2 = pca1;
% Z = transformed
kZ = 10;
aZ = 10;
pcaZ = 0;
% diffusion time for final embedding
dt = 1;
%
output = zeros(NPs, Ns, NWs, 2);  %store metrics in here
%%

for p = 1:NPs
    p
    %build random matrix and replace prct of columns with I
    pct = Ps(p)
    randomrot = orth(randn(784)); %random orthogonal rotation
    colReplace = randsample(784,floor(pct*784));
    randomrot(:,colReplace) = I(:,colReplace);
    
    for iter = 1:Ns
        iter
        rng('shuffle');

    % sample two sets of digits from MNIST
        rs1 = randsample(length(labels), N);
        rs2 = randsample(length(labels), N);
        
    % slice the digits
        x1 = imgs(rs1,:);
        x2b = imgs(rs2,:);
        
    % transform x2
        x2 = (x2b*randomrot');
        x3 = [x1;x2];
        [u3, v3, L3] = diffusionCoordinates(x3,a1,k1,pca1); %this is for evaluating unaligned data.  You can also ploit this.
    % slice the labels
        l1 = labels(rs1);
        l2 = labels(rs2);
        
    % run pca and classify
        DM3  = u3*diag(exp(-(v3)));
        beforeprct = knnclassifier(DM3(1:N,:), l1, DM3(N+1:end,:), l2, 5);
        
    % construct graphs 
        [u1, v1, L1] = diffusionCoordinates(x1,a1,k1,pca1); %normalized L with diffusion coordinates for sample 1
        [u2, v2, L2] = diffusionCoordinates(x2,a2,k2,pca2); %... sample 2
        
    % get fourier coefficients
        x1hat = u1'*x1;
        x2hat = u2'*x2;
        
        for scale = 1:NWs
            %iterate over bandwidths
            Nf = Ws(scale);
            
            % build wavelets
            [we1, ~, ~] = build_wavelets(v1,Nf,2);
            [we2, ~, ~] = build_wavelets(v2,Nf,2);
            %we1 is the filter evaluated over the eigenvalues.  So we can
            %pointwise multiply each we1/2 by the fourier coefficients
            
            
            % evaluate wavelets over data in the spectral domain
            % stolen from gspbox, i have no idea how the fuck this works
            c1hat = bsxfun(@times, conj(we1), permute(x1hat,[1 3 2]));
            c2hat = bsxfun(@times, conj(we2), permute(x2hat,[1 3 2]));
            
            % correlate the spectral domain wavelet coefficients.
            blocks = zeros(size(c1hat,1), Nf, size(c2hat,1));
            for i = 1:Nf %for each filter, build a correlation
                blocks(:,i,:) = (squeeze(c1hat(:,i,:))*squeeze(c2hat(:,i,:))');
            end
            % construct transformation matrix
            M = squeeze(sum(blocks,2)); %sum wavelets up 
            [Ut,St,Vt] = randPCA(M, min(size(M))); %this is random svd

            St = St(St>0); %this is here from earlier experiments where I was truncating by rank.  
            % We can probably remove this.
            rk = length(St);
            Ut = Ut(:,1:rk);
            Vt = Vt(:,1:rk);


            T = Ut*Vt'; %the orthogonal transformation matrix
            
            % compute transformed data
            
            u1T =  u1* (T) ; %U1 in span(U2)
            T = T';

            u2T = u2 * (T); % U2 in span(U1)
            
            E = [u1 u1T; u2T u2];
            
            X = E *diag(exp(-dt.*([v1;v2])));
            [uZ, vZ, LZ] = diffusionCoordinates(X, aZ, kZ, pcaZ);
            
            Z = uZ*diag(exp(-vZ));
            afterprct = knnclassifier(Z(1:N,:), l1, Z(N+1:end,:), l2, 5);
            output(p, iter, scale, 1) = beforeprct;
            output(p,iter, scale, 2) = afterprct;
        end
    end
     
end
%%

%%
function prct = knnclassifier(x, lx, y, ly, k) %check the nearest neighbors and their associated labels.
    nns = knnsearch(x, y, 'k', k);
    prct = ly == mode(lx(nns),2);
    prct = sum(prct)/length(prct);
end

function [u,v,L] = diffusionCoordinates(x,a,k,npca)
    %diffusion maps with normalized Laplacian
    %npca = 0 corresponds to NO pca
    [~, w] = alphakernel(x, 'a',a, 'k',k,'npca',npca);
    N = size(w,1);
    D = sum(w,2);
    w = w./(D*D'); %this is the anisotropic kernel
    D = diag(sum(w,1));
    L = eye(N)-D^-0.5 * w * D^-0.5;

    disp('svd L')
    [u,v,~] = randPCA(L,N);
    [ss,ix] = sort(diag(v));
    v = ss;
    u = u(:,ix);
    % trim trivial information
    u = u(:,2:end);
    v = v(2:end);
end

function [fe,Hk,mu] = build_wavelets(v,Nf,overlap)
    lmax = max(v); %maximum laplacian eigenvalue
    k = @(x) sin(0.5 * pi * (cos(pi*x)).^2) .* (x>=-0.5 & x<= 0.5);
    %this is the itersine function

    Hk = cell(Nf,1); %we are gonna store some lambda functions in here

    scale = lmax/(Nf-overlap+1)*(overlap);

    mu = zeros(Nf,1);
    % this is translating the wavelets along the interval 0, lmax.  
    for ii=1:Nf
        Hk{ii} = @(x) k(x/scale-(ii-overlap/2)/overlap)...
                    ./sqrt(overlap)*sqrt(2); %lambda functions for the spectral domain filters..
        mu(ii) = (ii-overlap/2)/overlap * scale; %i think this is the mean of each filter
    end
% response evaluation... this is the money 
    fe=zeros(length(v),Nf);
    for ii=1:Nf
        fe(:,ii)=Hk{ii}(v); 
    end
end
