function [H, P] = mvarx_residual_whiteness(W, varargin) 
%MVARX_RESIDUAL_WHITENESS residual whiteness test/consistency test 
%
% W - MVARX residaul - part of output of mvarx_fit, can be either a matrix
%	or a cell
% 	- W is an M-by-(N - n_o) matrix, where M is the number of channels,
%	  N is the number of samples in the data and n_o = max(p, l - 1)
%	- W is a 1-by-J cell with W{j} being a M-by-(N_j - n_o) matrix for j = 1, 2, ..., J
%
% [H, P] = mvarx_residual_whiteness(W)
% 	tests the residaul whiteness in W with Bartlett Window and Lag size L = ceil(3*N^0.3)
%   where N = max([N_1 - n_o, N_2 - n_o, ..., N_J - n_o])
%
% H is the hypothesis statistics under the hypothesis that the residual is uncorrelated
% P is the p-value
%
% [H, P] = mvarx_residual_whiteness(W, 'kernel', kernel_type, 'L', Lag)
%
% tests the residual whiteness with selected kernel type and Lag length
% Kernel types can be 'TR', 'BAR', 'DAN', 'PAR', and 'QS'
% L can be 2, 3, 'log', 'n_to_point2', and 'n_to_point3'

options = containers.Map({'kernel', 'L'}, {'BAR', 'n_to_point3'});    

% varargin not even
if mod(length(varargin) , 2) ~= 0
    error('mvarx_residual_whiteness needs propertyName/propertyValue pairs.');
end

for pair = reshape(varargin, 2, [])   % pair is {propName; propValue}
    inpName = pair{1};

    if options.isKey(inpName)
        options(inpName) = pair{2};
    else
        error('%s is not a recognized parameter name.', inpName);
    end
end

assert(any(strcmp({'TR', 'BAR', 'DAN', 'PAR', 'QS'}, options('kernel'))), ...
	'ERROR: Selected kernel type not supported.');

assert(any(strcmp({'log', 'n_to_point2', 'n_to_point3', '2', '3'}, options('L'))), ...
	'Error: Given value of L not supported.');

if iscell(W)
	d = size(W{1}, 1);
	n_epoch = size(W, 2);
	epoch_sizes = cellfun(@(x) size(x, 2), W);

	% longest correlation lag would be the number of residual samples in the 
	% longest epoch minus 1
 	corr_window = max(epoch_sizes) - 1;
else
	d = size(W, 1);
	n_epoch = 1;
	epoch_sizes = size(W, 2);

	% longest correlation lag would be the number of residual samples minus 1
	corr_window = epoch_sizes - 1;

	W = {W};
end

corr_mat = zeros(d, d, corr_window + 1);
for i = 0:corr_window
	for j = 1:n_epoch
		corr_mat(:, :, i+1) = corr_mat(:, :, i+1) + (W{j}(:, 1:end-i) * W{j}(:, i+1:end).');
	end
end
corr_mat = corr_mat / sum(epoch_sizes);
corr_mat = num2cell(corr_mat, [1 2]);

krn_type = options('kernel');

n_spl = max(epoch_sizes);
switch options('L')
case '2'
    L = 2;
case '3'
    L = 3;
case 'log'
    L = ceil(log(n_spl));
case 'n_to_point2'
    L = ceil(3.5 * n_spl^0.2);
case 'n_to_point3'
    L = ceil(3 * n_spl^0.3);
end

kz = ker_fn(krn_type, corr_window, L)';

M_n = (1 - (1:n_spl-1) / n_spl) * kz.^2;
V_n = ((1 - (1:n_spl-2) / n_spl) .* (1 - (2:n_spl-1) / n_spl)) * kz(1:n_spl-2).^4; 

tmp = 0;
for j = 1:corr_window
    tmp = tmp + kz(j)^2 * trace(corr_mat{j+1}.'/corr_mat{1}*corr_mat{j+1}/corr_mat{1});
end
T_n = (n_spl * tmp - d^2*M_n) / sqrt(2*d^2*V_n);

H = T_n;
P = 2 * (1 - normcdf(abs(H)));