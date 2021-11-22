function [U_final, V_final, obj] = TV_CNMF(X, k, W, options, U, V)
% Total Variation Constrained Graph-Regularized Convex Non-Negative ...
% Matrix Factorization (TV-GCNMF)

% Notation:
% X ... (mFea x nSmp) data matrix
%       mFea  ... number of features
%       nSmp  ... number of samples
% k ... number of hidden factors
% W ... weight matrix of the affinity graph
%
%
% options ... Structure holding all settings
% options.maxIter ... Max Iterations
% options.lambda  ... hyper parameter lambda
% options.beta    ... hyper parameter beta
%
% You only need to provide the above four inputs.
%
% References:
%[1] M. Tian, C. Leng, H. Wu, and A. Basu, "Total Variation Constrained
%    Graph-Regularized Convex Non-Negative  Matrix Factorization
%    for Data Representation," IEEE Signal Processing Letters, vol.28, pp.
%    126-130, 2020.
%
% Written by Haonan Wu (fancrey@gmail.com or fancrey@stumail.nwu.edu.cn)
% This file is under GNU General Public License, version 3 (GPL-3.0)
% github: https://github.com/Fancrey/TV_GCNMF

if min(min(X)) < 0
    error('Input should be nonnegative!');
end

if ~isfield(options, 'maxIter')
    options.maxIter = 500;
end

U = [];
V = [];

K = constructKernel(X',[],options);
[U_final, V_final, obj] = TV_CNMF_Multi(K, k, W, options, U, V);
end



