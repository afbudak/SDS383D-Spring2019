%%%%%%%%%%%%%%      Linear Smoothing        %%%%%%%%%%%%%%%

%start with generating some data to smooth
clear;clf;
X = (-20:0.1:20);
mu = 0;
sigma = 1;
y = sin(X) + normrnd(mu, sigma, 1, length(X));
plot(X, y);
gaussiankernel = @normpdf;
xstar = -20:0.01:20;
h = 1;
m = length(xstar);

for i = 1:m
    ystar(i) = predict(X, y, xstar(i), h, gaussiankernel);
end

hold on;
plot(xstar,ystar);


function ystar = predict(X, y, xstar, h, kerneltype)
    w = weight(X, xstar, h, kerneltype);
    ystar = sum( w .* y );
end


function w = weight(X, xstar, h, kerneltype)
    n = length(X);
    for i = 1:n
        w_temp(i) = 1 / h * kerneltype( (X(i) - xstar) /h );
    end
    w = w_temp / sum(w_temp);
end

        

