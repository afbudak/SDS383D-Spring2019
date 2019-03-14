%%%%%%%%%%%%%   Cross Validation    %%%%%%%%%%%%%%

%The first part is kernel fitting
clear;
l = -5; r = 5; n = 100; std = 0.4;
[X, y] = generatedata(n, std, l, r, @sin);
scatter(X, y);
gaussian = @normpdf;
h = 0.8;
[xstar, ystar] = fit_kernel_smoother(X, y, l, r, h, gaussian);
hold on;
plot(xstar, ystar);

%%%%%Cross Validation%%%%%
h_trial = linspace(0, 1);
num_h = length(h_trial);
%Use sin function as wiggle function
for i = 1:num_h
    h = h_trial(i);
    [Xtest, ytest] = generatedata(n, std, l, r, @sin);
    n = length(Xtest);
    ystar = zeros(1,n);
    for j = 1:n
        ytest_star(j) = predict(X, y, Xtest(j), h, gaussian);
    end
    mse(i) = sqrt( (ytest-ytest_star) * (ytest-ytest_star)' );
end
[~, idx] = min(mse);
min_htiral_wiggle = h_trial(idx);

%use f(x) = x^2 as smooth function
[X, y] = generatedata(n, std, l, r, @fsmooth);
figure
scatter(X, y);
gaussian = @normpdf;
h = 0.8;
[xstar, ystar] = fit_kernel_smoother(X, y, l, r, h, gaussian);
hold on;
plot(xstar, ystar);
for i = 1:num_h
    h = h_trial(i);
    [Xtest, ytest] = generatedata(n, std, l, r, @fsmooth);
    n = length(Xtest);
    ystar = zeros(1,n);
    for j = 1:n
        ytest_star(j) = predict(X, y, Xtest(j), h, gaussian);
    end
    mse(i) = sqrt( (ytest-ytest_star) * (ytest-ytest_star)' );
end
[~, idx] = min(mse);
min_htiral_smooth = h_trial(idx);

%Notice that best h value for smooth function is much smaller than the
%wiggly function.

%% functions used above
function res = fsmooth(x)
    res = x.^2;
end

function [xstar, ystar] = fit_kernel_smoother(X, y, l, r, h, kernel)
    xstar = linspace(l,r,1000);
    n = length(xstar);
    ystar = zeros(1,n);
    for i = 1:n
        ystar(i) = predict(X, y, xstar(i), h, kernel);
    end
end

function res = predict(X, y, xstar, h, kernel)
    w = weight(X, xstar, h, kernel);
    res = w*y';
end

%Define weight
function w = weight(X, xstar, h, kernel)
    n = length(X);
    for i = 1:n
        d = X(i) - xstar;
        w(i) = 1/h * kernel(d/h);
    end
    w = w/sum(w);                %normalize w values
end

function [X, y] = generatedata(n, std, l, r, func)
    X = l + (r-l)*rand(1,n);
    y = func(X) + std*randn(1,n);
end
