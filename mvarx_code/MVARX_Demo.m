
% load MVARX models estimated from Chang, et al. Front Hum Neurosci. 2012; 6: 317. 
load('mdl_cltd.mat', 'mdl_F3_FA_5m')

m_ori = size(mdl_F3_FA_5m.Aw, 1);    % number of channel in the original model
m = 3;    % number of channels we will be using in this simulation
p = 3;    % MVARX AR order
l = 0;   % MVARX feedforward length

col_set = kron(ones(1, m), 1:p) + kron(0:m_ori:m_ori*(p-1), ones(1, m));
A = mdl_F3_FA_5m.Aw(1:m, col_set);     % MVARX A matrix
B = mdl_F3_FA_5m.Bw(1:m, 1:l);         % MVARX B matrix
Q = 25 * mdl_F3_FA_5m.Qw(1:m, 1:m);    % MVARX Q matrix

while ~is_stbl(A)
    A = A * 0.9;
end

u = [zeros(1, 19), 1, zeros(1, 80)];  % train of stimulation

Y = mvarx_data_gen(A, B, Q, u);

figure
wf_shift = (0:-20:(m-1)*(-20))';
plot((Y + wf_shift(:, ones(1, 100)))', 'Color', [31,120,180] / 255);
set(gca, 'ytick', '')
xlabel('t')

n_epoch = 20;
Y = cell(1, n_epoch);
for i = 1:n_epoch
    Y{i} = mvarx_data_gen(A, B, Q, u);
end

% create a 1-by-20 cell, each cell is the train of stimulation for the epoch
u = num2cell(repmat(u, 1, 1, n_epoch), [1, 2]);   

p = 3;
l = 0;% this is coding for duration of exogenous effect. Don't worry about this for now.

[A_hat, B_hat, Q_hat, W, n_spl] = mvarx_fit(Y, u, p, l);

cmin = min([A_hat(:); A(:)]);
cmax = max([A_hat(:); A(:)]);
subplot(211); imagesc(A, [cmin, cmax]); title('$A$', 'Interpreter', 'latex'); colorbar;
subplot(212); imagesc(A_hat, [cmin, cmax]); title('$\hat A$', 'Interpreter', 'latex'); colorbar;

%evoked_response = mean(reshape(cell2mat(Y), m, size(u{1},2), []), 3);
% model_response = mvarx_data_gen(A_hat, B_hat, [], u{1}, 'evoked', true);

% wf_shift = (0:-20:(m-1)*(-20))';
% h1 = plot((evoked_response + wf_shift(:, ones(1, 100)))', 'Color', [31,120,180] / 255); hold on;
% h2 = plot((model_response + wf_shift(:, ones(1, 100)))', 'Color', [227,26,28] / 255);
% legend([h1(1), h2(1)], {'Evoked Response', 'Model Response'})
% 
% [H, p] = mvarx_residual_whiteness(W{1})
% 
% [H, p] = mvarx_residual_whiteness(cell2mat(W), 'kernel', 'PAR', 'L', 'log')
