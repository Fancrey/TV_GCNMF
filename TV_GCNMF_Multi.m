function [U_final, V_final, obj] = TV_CNMF_Multi(K, k, W, options, U, V)
% Total Variation Constrained Graph-Regularized Convex Non-Negative ...
% Matrix Factorization (TV-GCNMF)


% Notation:
% K ... (nSmp x nSmp) kernel matrix
% k ... number of hidden factors
% W ... weight matrix of the affinity graph
%
% options ... Structure holding all settings
% options.maxIter ... Max Iterations
% options.lambda  ... hyper parameter lambda
% options.beta    ... hyper parameter beta
% obj             ... objective functions with every iterations
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

maxIter = options.maxIter;
beta=options.beta;
lambda = options.lambda;
Norm = 2;
NormV = 1;
nSmp = size(K,1);
obj=zeros(1,maxIter);

if lambda < 0
    lambda = 0;
end
W = lambda*W;
DCol = full(sum(W,2));
D = spdiags(DCol,0,speye(size(W,1)));
L = D - W;
if isempty(U)
    U = abs(rand(nSmp,k));
    V = abs(rand(nSmp,k));
end
[U,V] = NormalizeUV(K, U, V, NormV, Norm);

nIter=1;
while(nIter <= maxIter)
    % ===================== update V ========================
    KU = K*U;
    UKU = U'*KU;
    VUKU = V*UKU;
    [ny,nx]=size(V); ep=1.2;ep2=ep^2;
    I_x = (V(:,[2:nx nx])-V(:,[1 1:nx-1]))/2;
    I_y = (V([2:ny ny],:)-V([1 1:ny-1],:))/2;
    I_xx = V(:,[2:nx nx])+V(:,[1 1:nx-1])-2*V;
    I_yy = V([2:ny ny],:)+V([1 1:ny-1],:)-2*V;
    Dp = V([2:ny ny],[2:nx nx])+V([1 1:ny-1],[1 1:nx-1]);
    Dm = V([1 1:ny-1],[2:nx nx])+V([2:ny ny],[1 1:nx-1]);
    I_xy = (Dp-Dm)/4;
    Num = I_xx.*(ep2+I_y.^2)-2*I_x.*I_y.*I_xy+I_yy.*(ep2+I_x.^2);
    Den = (ep2+I_x.^2+I_y.^2).^(3/2);
    I_t = Num./Den;
    
    if lambda > 0
        WV = W*V;
        DV = repmat(DCol,1,k).*V;      
        KU = KU + WV +beta*I_t;
        VUKU = VUKU + lambda*DV;
    end
    V = V.*(KU./max(VUKU,1e-10));
    clear WV DV KU UKU VUKU;
    % ===================== update U ========================
    KV = K*V;             
    VV = V'*V;            
    KUVV = K*U*VV;          
    U = U.*(KV./max(KUVV,1e-10));
    clear KV VV KUVV;
    
    [obj_NMFhistory, obj_Laphistory] = CalculateObj(K, U, V, L, I_x, I_y);
    obj(nIter)=obj_NMFhistory+obj_Laphistory;
    nIter = nIter + 1;
end
[U_final,V_final] = NormalizeUV(K, U, V, NormV, Norm);

%==========================================================================

function [obj_NMF, obj_Lap] = CalculateObj(K, U, V, L, I_x, I_y)

UK = U'*K; 
UKU = UK*U; 
VUK = V*UK; 
VV = V'*V; 
obj_NMF = sum(diag(K))-2*sum(diag(VUK))+sum(sum(UKU.*VV));
obj_Lap = sum(sum((L*V).*V))+norm((abs(I_x)+abs(I_y)),'fro');



function [U, V] = NormalizeUV(K, U, V, NormV, Norm)
k = size(U,2);
if Norm == 2
    if NormV
        norms = max(1e-15,sqrt(sum(V.^2,1)))';
        V = V*spdiags(norms.^-1,0,k,k);
        U = U*spdiags(norms,0,k,k);
    else
        norms = max(1e-15,sqrt(sum(U.*(K*U),1)))';
        U = U*spdiags(norms.^-1,0,k,k);
        V = V*spdiags(norms,0,k,k);
    end
else
    if NormV
        norms = max(1e-15,sum(abs(V),1))';
        V = V*spdiags(norms.^-1,0,k,k);
        U = U*spdiags(norms,0,k,k);
    else
        norms = max(1e-15,sum(U.*(K*U),1))';
        U = U*spdiags(norms.^-1,0,k,k);
        V = V*spdiags(norms,0,k,k);
    end
end


