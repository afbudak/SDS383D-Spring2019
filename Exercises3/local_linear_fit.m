clear;
data = csvread('utilities.csv',1,0);
x = data(:,1);
y = data(:,2)./data(:,3);           %bill per day
y = y';
scatter(x, y);

%get training and testing data differently
allindexes = 1:length(x);
trainindexes = 1:2:length(x);
idx = ismember(allindexes,trainindexes);
xtrain = x(idx); ytrain = y(idx); 
xtest = x(~idx); ytest = y(~idx);
%xstar = sort(x);

kernelfunction = @normpdf;  D = 1;

%experiment for h minimizing the error
h_trial = linspace(1,20,100);
for i = 1:length(h_trial)
    h = h_trial(i);
    for j = 1:length(xtest)
        [~, yteststar(j)] = predict_loc_lin(xtrain, ytrain, xtest(j), h, kernelfunction, D);
    end
    mse(i) = sqrt( (yteststar - ytest)*(yteststar - ytest)' );
end

%choose h with min mse
[~, idx] = min(mse);
hbest = h_trial(idx);
[xtest, order] = sort(x);
yordered = y(order);
n = length(xtest);
Herror = zeros(n);
for j = 1:n
    [H, yteststar(j)] = predict_loc_lin(x, y, xtest(j), hbest, kernelfunction, D);
    Herror(j,:) = H(1,:);
end
hold on;
plot(xtest, yteststar);
xlabel('temp');
ylabel('average daily bill');
figure;
plot(h_trial, mse);
xlabel('h sweep');
ylabel('~mse');

errors = yordered - yteststar;
figure
scatter(xtest,errors);

%Calculate Confidence Interval
rsquared = errors*errors';
sigmasquared = rsquared / ( n - 2*trace(Herror) + trace(Herror.'*Herror) );
%MATLAB does not have a proper plotting for CI (to my knowledge)

%% Related functions
%prediction with local linear kernel

function [H, ystar] = predict_loc_lin(X, y, xstar, h, kernelfunction, D)
    %define dimensions and matrixes
    n = length(y);
    R = zeros(n, D+1);
    K = zeros(n);
    for i = 1:n
        K(i,i) = 1/h * kernelfunction( (X(i) - xstar) / h );
        R(i,1) = 1;
        for j = 2:D+1
            R(i,j) = (X(i) - xstar)^(j-1);
        end
    end
    H = ( transpose(R) * K * R ) \ transpose(R) * K;
    coeffs = H*y';                  %poly coeffs  
    ystar = coeffs(1);
end

function error = loc_lin_error(X, y, h, kernelfun, D)
    n = length(X);
    error = 0;
    for i = 1:n
        xstar = X(i);
        [a, H] = predict_loc_lin(X, y, xstar, h, kernelfun, D);
        error = error + ( (y(i) - a(1)) / (1 - H(i)) ) ^ 2;  
    end
end

%calculate ystar for all xstar
function ystar = predict_llk_all(X, y, h, kernelfun, D)
    n = length(X);
    for i = 1:n
        xstar = X(i);
        [~, ystar(i)] = predict_loc_lin(X, y,  xstar, h, kernelfun, D);
    end
end

function H = get_hat_matrix(X, y, h, kernelfun, D)
    n = length(X);
    for i = 1:n
        xstar = X(i);
        [~, H{i}] = predict_loc_lin(X, y, xstar, h, kernelfun, D);
    end
end

