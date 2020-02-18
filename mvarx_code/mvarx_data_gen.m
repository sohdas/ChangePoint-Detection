function [Y, Ryy, Ryz, Rzz] = mvarx_data_gen(A, B, Q, u, varargin)
%MVARX_DATA_GEN generate MVARX simulation data/evoked response
% 
% generate MVARX simulated data that satsify the following relation
% X(:, n) = A * [X(:, n - 1); X(:, n - 2); X(:, n - 3); ... ; X(:, n - p)]
%          + B * [u(n); u(n - 1); ... ; u(n - l + 1)] + W(:, n - n_o)
%         where [X(:, 0); X(:, -1); X(:, -2); ... ; X(:, - p + 1)] ~ N(0, Sigma_0)
%         and Sigma_0 = dlyap(As, Qs)  (discrete Lyapunov equations)
%         with As = [A; eye(M * (p - 1)), zeros(M * (p - 1), M))] and
%         Qs = blkdiag(Q, eye(M * (p - 1)))
% 
% If Q = [] then genreate Evoked Response
%
% A - MVARX autoregressive coefficient matrix (M-by-Mp)
% B - MVARX direct feedforward matrix (M-by-l)
% Q - MVARX residual covariance mattrix (M-by-M)
% u - Stimulation sequence (1-by-N) representing the application of stimulation in time
%
% Y = mvarx_data_gen(A, B, Q, u)  
% generate an epoch of MVARX time series with coefficients A, B, and Q
% and a train of stimulation u 
%
% Y = mvarx_data_gen(A, B, [], u, 'evoked', true)
% generate an epoch of MVARX model evoked response with coefficients A and B
% and a train of stimulation u
%
% Y = mvarx_data_gen(A, B, Q, u, 'baseline', mu)
% generate an epoch of MVARX time series of baseline (mean of time series) mu
% with coeffiicents A, B, and Q and a train of stimulation u 
%
% Y = mvarx_data_gen(A, B, [], u, 'evoked', true, 'baseline', mu)
% generate an epoch of MVARX model evoked responseof baseline mu with coefficients A and B
% and a train of stimulation u

mem_size = size(A, 2);
M = size(A, 1);
p = mem_size / M;
l = size(B, 2);
n_spl = length(u);

options = containers.Map({'evoked', 'baseline'}, {false, zeros(M, 1)});    

% varargin not even
if mod(length(varargin) , 2) ~= 0
    error('mvarx_data_gen needs propertyName/propertyValue pairs.');
end

for pair = reshape(varargin, 2, [])   % pair is {propName; propValue}
    inpName = lower(pair{1});        % make case insensitive

    if options.isKey(inpName)
        if strcmp(inpName, 'baseline')
            assert(length(pair{2}) == M, ...
                'ERROR: The baseline must be of the same dimension as the number of channels in A.');
        end
        options(inpName) = pair{2};
   else
      error('%s is not a recognized parameter name', inpName)
   end
end

% extract the options
%for k = options.keys
%   eval([k{1}, ' = options(''', k{1}, ''');'])
%end

% check if A is not stable
if ~is_stbl(A)
    fprintf('A is not stable.\n')
end

As = [A; eye(mem_size - M), zeros(mem_size - M, M)];
tmp = (eye(mem_size) - As) * repmat(options('baseline'), p, 1);
c = tmp(1:M);

if options('evoked')
    % generating model evoked response
    Z0 = zeros(mem_size, 1) + repmat(options('baseline'), p, 1);
    W = zeros(M, n_spl);
else
    % simulating mvarx time series
    Qs = blkdiag(Q, zeros(mem_size - M));
    init_cov = dlyap(As, Qs);

    Z0 = chol(init_cov).' * randn(mem_size, 1) + repmat(options('baseline'), p, 1);
    W = chol(Q).' * randn(M, n_spl);
end

if ~isempty(B)
    D = toeplitz([u(1); zeros(l-1,1)], u);
else
    D = [];
end

Z = [Z0, zeros(mem_size, n_spl-1); D];

Y = zeros(M, n_spl);
theta = [A, B];
for i = 1:n_spl-1
    Y(:, i) = theta*Z(:, i) + W(:, i) + c;
    Z(1:M*p, i+1) = [Y(:, i); Z(1:M*(p-1), i)];
end

if ~isempty(i)
    Y(:, i+1) = theta*Z(:, i+1) + W(:, i+1) + c;
else
    Y(:, 1) = theta*Z(:,1) + W(:, 1) + c;
end

Ryy = Y*Y.';
Ryz = Y*Z.';
Rzz = Z*Z.';