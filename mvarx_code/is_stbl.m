function [stbl,lambda] = is_stbl(A)
%IS_STBL check stability of AR coefficient of MVARX model
% stbl = is_stbl(A)
% 
% A - MVARX autoregressive coefficient
% 
% stbl - logical variable, stbl = 1 if A is stable, stbl = 0 if A is unstable
M = size(A, 1);
p = size(A, 2) / M;

% state space transition matrix
A_s = [A; eye(M * (p - 1)), zeros(M * (p - 1), M)];

lambda = max(abs(eig(A_s)));

stbl = (lambda < 1);