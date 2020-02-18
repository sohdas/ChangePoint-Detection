function [A, B, Q, W, n_spl] = mvarx_fit(X, u, p, l)
%MVARX_FIT Fit MVARX model to data
% [A, B, Q, W, n_spl] = mvarx_fit(X, u, p, l)
%
% Find an MVARX model that fits the data with the following relation
% X(:, n) = A * [X(:, n - 1); X(:, n - 2); X(:, n - 3); ... ; X(:, n - p)]
%          + B * [u(n); u(n - 1); ... ; u(n - l + 1)] + W(:, n - n_o)
%          for n = n_o + 1, ..., N where n_o = max(p, l - 1)
%
% X - Data, can be either a matrix or a cell
%       -if X is an M-by-N matrix, X would be the measurements from M 
%        channels/electrodes in a window of N samples
%       -if X is a 1-by-J cell, each cell is measurements from M channels in
%        an epoch/trial, X{j} is an M-by-N_j matrix for j = 1, 2, ..., J
%        assuming there are J epochs
% u - stimulation
%       -if X is a matrix, u should be a 1-by-N vector, representing the application
%        of stimulation in time
%       -if X is a cell, u should also be a 1-by-J cell, and u{j} is an 1-by-N_j
%        vector for j = 1, 2, ..., J
% p - MVARX model autoregressive order
% l - MVARX model direct feedforward effect length
%
% A - MVARX autoregressive coefficient matrix (M-by-Mp)
% B - MVARX direct feedforward matrix (M-by-l)
% W - MVARX residual 
%       -if X is a matrix then W will be an M-by-(N - n_o) matrix 
%       -if X is a cell then W wil be a 1-by-J cell, with each cell W{j} being 
%        an M-by-(N_j - n_o) matrix
% Q - MVARX residual covariance matrix E[Q] = E[W(:, n)* W(:, n).'] for all n
% n_spl - number of samples in data
%       -if X is a matrix then n_spl is a scalar and n_spl = N - n_o
%       -if X is a cell then n_spl is a cell and n_spl{j} = N_j - n_o for j = 1, 2, ..., J

n_o = max(p, l - 1);

if ~iscell(X)
    % X is a matrix
    Data = X;

    [M, N] = size(Data);
    X = Data(:, n_o + 1:end);
    % Z is the design matrix
    Z = zeros(M * p + l, N - n_o);
    for i = 1:p
    	Z((i - 1) * M + 1:i * M, :) = Data(:, n_o + 1 - i:end - i);
    end

%     D = toeplitz([u(1); zeros(l - 1, 1)], u);
%     Z(p * M + 1:end, :) = D(:, n_o + 1:end);

    % solve the least squares problem
    theta = (X*Z.') / (Z*Z.');
    A = theta(:, 1:M * p);
    B = theta(:, M * p + 1:end);

    W = (X - theta * Z);
    Q = (W * W.') / (N - n_o);
    n_spl = N - n_o;
else
    % X is a cell
    Data = X;
    n_epoch = size(Data, 2);    % number of trials/epochs
    M = size(Data{1}, 1);       % channels

    R_XZ = zeros(M, M * p + l);
    R_ZZ = zeros(M * p + l);

    Data_cltd = cell(1, n_epoch);
    Z_cltd = cell(1, n_epoch);
    n_spl = cell(1, n_epoch);

    for j = 1:n_epoch
        n_spl{j} = size(Data{j}, 2) - n_o;

    	X = Data{j}(:, n_o + 1:end);
        Z = zeros(M * p + l, n_spl{j});

        for i = 1:p
            Z((i - 1) * M + 1:i * M, :) = Data{j}(:, n_o + 1 - i:end - i);
        end
        if l > 0
            D = toeplitz([u{j}(1); zeros(l - 1, 1)], u{j});
            Z(M * p + 1:end, :) = D(:, n_o + 1:end);
        end
        R_XZ = R_XZ + (X*Z.');
    	R_ZZ = R_ZZ + (Z*Z.');

    	Data_cltd{j} = X;
    	Z_cltd{j} = Z;
    end

    theta = R_XZ / R_ZZ;  
    A = theta(:, 1:M * p);   
    B = theta(:, M * p + 1:end);

    W = cell(1, n_epoch);
    for j = 1:n_epoch
        W{j} = Data_cltd{j} - theta * Z_cltd{j};
    end
    Q = (cell2mat(W)*cell2mat(W).') / sum(cell2mat(n_spl));   
end