%heavy tailed method
clear;
data = csvread('gdpgrowth.csv',1,2);
y = data(:,1);                          %growth rate
X = data(:,6);                          %defence spending
X = [ones(length(X),1), X];
n = length(y);                          %number of sample
p = size(X,2) - 1;                      %number of variable
d = 0.01; eta = 0.01; h = 0.1;
m = zeros(p+1, 1); 
K = 0.005*diag(p);

%Gibbs sampling phase
niter = 12000;
[betas_post, lambdas_post, omegas_post] = heavytailedGibbs(X, y, m, K, d, eta, h, niter);
betameans = mean(betas_post);
ystar = betameans(1) + betameans(2)*X(:,2);

%Plotting
scatter(X(:,2),y);
hold on;
plot(X(:,2),ystar);

%Compare with the linear model
linfitmodel = X\y; 
ystarlin = linfitmodel(1) + linfitmodel(2)*X(:,2);
hold on;
%Plotting and Labeling
plot(X(:,2), ystarlin);
xlabel('Defense Spending');
ylabel('GDP Rate');
legend('data','Heavy Tailed Method','Simple Linear Model');

function [betas, lambdas, omegas] = heavytailedGibbs(X, y, m, K, d, eta, h, niter)
    n = length(y);                          %number of sample
    p = size(X,2) - 1;                      %number of variable
    betas = zeros(niter, p+1);
    lambdas = zeros(niter, n);
    omegas = zeros(1, niter);
    
    %Initialization
    betas(1,:) = 0;
    lambdas(1,:) = 2;
    omegas(1) = 2;
    
    for i = 1:(niter-1)
        Lambda = diag(lambdas(i,:));
        prod1 = K + transpose(X)*Lambda*X;
        prod2 = K*m + transpose(X)*Lambda*y ;
        mu_star = prod1\prod2;                  %multiplying with inv of prod1
        d_star = d + n + p;
        eta_star = transpose(y)*Lambda*y + transpose(m)*K*m + eta - ...
                    transpose(prod2)*(prod1\prod2);
        betas(i+1,:) = mvnrnd(mu_star, inv(prod1)/omegas(i));
        omegas(i+1) = gamrnd(d_star/2, 1/(eta_star/2));
        lambda_rate = 0.5 * (omegas(i+1) * ( y - X*betas(i+1,:)' ).^2 + h);
        lambdas(i+1,:) = gamrnd((h+1)/2*ones(1,n), 1./lambda_rate');     
    end
end