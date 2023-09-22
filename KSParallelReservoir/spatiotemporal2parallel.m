clear all;

n_data = 64; n_steps = 80000;
% data_kind = 'NLCGL';
data_kind = 'CGL';
L = 8; N = 32; dps = n_steps;
% load(['\\nas08c093\data\otsuki\spatiotemporal\results\' data_kind '1d\Adata_L18_N_32_dps', num2str(n_steps), '.mat']);
load(['\\nas08c093\data\otsuki\spatiotemporal\results\' data_kind '1d\Adata_L8_N_32_dps', num2str(n_steps), '.mat']);

Adata = [real(Adata(:,1:end-1)); imag(Adata(:,1:end-1))].';
train_input_sequence = zeros(n_steps, n_data);
train_input_sequence(:, 1:2:end) = Adata(:, 1:n_data/2);
train_input_sequence(:, 2:2:end) = Adata(:, n_data/2+1:end);

clear Adata;

filename = [data_kind '_L' num2str(L) '_N_' num2str(N) '_dps' num2str(dps) '.mat'];
save(filename, '-v7.3');


clear all;
n_data = 64; n_steps = 20000;
% data_kind = 'NLCGL';
data_kind = 'CGL';
L = 8; N = 32; dps = n_steps;
% load(['\\nas08c093\data\otsuki\spatiotemporal\results\' data_kind '1d\Adata_L18_N_32_dps', num2str(n_steps), '.mat']);
load(['\\nas08c093\data\otsuki\spatiotemporal\results\' data_kind '1d\Adata_L8_N_32_dps', num2str(n_steps), '.mat']);

Adata = [real(Adata(:,1:end-1)); imag(Adata(:,1:end-1))].';
test_input_sequence = zeros(n_steps, n_data);
test_input_sequence(:, 1:2:end) = Adata(:, 1:n_data/2);
test_input_sequence(:, 2:2:end) = Adata(:, n_data/2+1:end);

clear Adata;

filename = [data_kind '_L' num2str(L) '_N_' num2str(N) '_dps' num2str(dps) '.mat'];
save(filename, '-v7.3');