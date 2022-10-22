function [P,G,W,B,objvalues] = LSAH(X,S,c,r, lambda1,lambda2,lambda3, maxiter)
% X n*d : data matrix 
% S n*m : anchor graph Z 
% L m*m : Laplacian of anchor point 
% c :     the number of classes 
% r :     r bit of B
rou = 1.05;
mu  = 0.0001;
[n, d] = size(X);
m = size(S,2);
Omega  = ones(m, c);
B=sign(randn(m,r));
U = eye(d,d);
P = rand(d,r); 
W = rand(r, c);   %% A == W
G = eye(m,c);   % F == G
Z = G;
SimlarityW = S' * S;
SimlarityD = diag(sum(SimlarityW,2));
L = SimlarityD - SimlarityW;
AL = lambda1*L;
XTX = X'*X;
for iter = 1:maxiter    
    %step 1: updata matrix W
    svdW = 2 * lambda2 * B' * G;
    [WU, ~, WV] = mySVD(svdW);  
    W = WU * WV';    
    %step 2 : updata matrix P 
% % %     invP = inv;
    P = (XTX+lambda3 * U) \ X' * S * B;    
    %step 3 update G, Z
    tempG = 2*lambda2*B*W - AL*Z+mu*Z-Omega;
    [U, ~, V] = mySVD(tempG);
    G = U*V';     
    muG = mu * G;
    Z = G - AL*G/mu + Omega/mu;
    Z(Z<0) = 0;
    muZ = mu * Z;   
    %step 4 update B
    SignB = 2 * S' * X * P + 2*lambda2 * G * W';
    B = sign(SignB);    
    %step 5 update U
    for i=1:d
        U(i,i) = 1 / (2 * norm(P(i,:),2) + eps) ;
    end 
    %step 6 update regualr parameters
    Omega  = Omega + muG - muZ;
    mu = min(10^10, rou*mu);
    objvalues(iter,1) = 1;
end
end
