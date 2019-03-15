%Gaussian LinearModel
clear;
data = csvread('gdpgrowth.csv',1,2);
y = data(:,1);                          %growth rate
X = data(:,6);                          %defence spending
X = [ones(length(X),1), X];
n = length(y);                          %number of sample
p = size(X,2) - 1;                      %number of variable
Lambda = eye(n);
d = 0.01; eta = 0.01;
m = zeros(p+1, 1); 
K = 0.005*eye(p);
prod1 = K + transpose(X)*Lambda*X;
prod2 = K*m + transpose(X)*Lambda*y ;
mu_star = prod1\prod2;                  %multiplying with inv of prod1
scatter(X(:,2), y);
X_fit = linspace(0,0.18);
y_fit = mu_star(1) + mu_star(2)*X_fit;
hold on;
plot(X_fit, y_fit);
title('linear gaussian fit');
xlabel('Defense Spending');
ylabel('GDP Growth');

