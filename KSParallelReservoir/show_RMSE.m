
% train_steps_list = [2e4+1 4e4:2e4:14e4];
% n_grids = size(train_steps_list, 2);
% rho = 1.6;
% rho = 0.6; 
% locality_list = [3 4 5 6 7 8];
locality_list = 18;
% rho_list = [1.1 1.2 1.3 1.4 1.5 1.6];
rho_list = 0.6;

data_kind = 'KS';
% data_kind = 'CGL';
data_kind = 'linear';
% input_kind = 'decay';
% input_kind = 'uniform';
max_lyapunov = 0.0743;
% n_steps = 18999;
% n_steps = 3798;
n_steps = 3000;

data_dir = '';
% locality_list = [1 2 3 4 5 6 7 8 12 16];
n_grids = size(locality_list, 2);

index_file = matfile([data_dir 'testing_ic_indexes.mat']);
pred_marker_array = index_file.testing_ic_indexes(1, 1:50);

approx_reservoir_size = 5000; locality = 8; num_workers = 8;
% n_data = 64; dt = 0.07; times = repmat(0:dt*max_lyapunov:(n_steps-1)*dt*max_lyapunov, n_data, 1);
n_data = 128; dt = 0.25; times = repmat(0:dt*max_lyapunov:(n_steps-1)*dt*max_lyapunov, n_data, 1);
errors = zeros(n_grids, n_steps);
figure();
for k = 1:n_grids
    % train_steps = train_steps_list(k);
    locality = locality_list(k);
    
    L = 50; N = 128;
    % L = 44; N = 64;
    % filename = ['.//' data_kind '/' data_kind 'result_decay_train80000_node7000-L50-radius0.6-locality' num2str(locality) '-numlabs8-jobid1-index_iter' num2str(1) '.mat'];
    filename = ['.//' data_kind '/' data_kind 'result_' input_kind '_train80000_node5000-L' num2str(L) '-radius0.6-locality' num2str(locality) '-numlabs8-jobid1-index_iter' num2str(1) '.mat'];
    % filename = [data_dir '/', data_kind, '/', data_kind, 'result_train', num2str(train_steps), '_node', num2str(approx_reservoir_size) '-L' num2str(L) '-radius' num2str(rho) '-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter', num2str(which_index_iter) '.mat'];
    % filename = ['.//NLKS/NLKSresult' num2str(j) '_train80000_node7000-L50-radius0.6-locality' num2str(locality) '-numlabs8-jobid1-index_iter' num2str(j) '.mat'];
    load(filename, 'error', 'resparams');
    pred_length = resparams.predict_length;
    for l = 1:length(pred_marker_array)
        errors(k, :) = errors(k, :) + error(1, (l-1)*pred_length+1:(l-1)*pred_length+n_steps);
    end
    errors(k, :) = errors(k, :) / length(pred_marker_array);

    hold on;
    % plot(error);
    plot(times(1,:), errors(k, 1:n_steps), 'DisplayName', ['locality=' num2str(locality)]); 
    % plot(times(1,:), error(1, 1:n_steps), 'DisplayName', ['train steps=' num2str(train_steps)]); 
    hold off;
end

% sgtitle(['L=' num2str(L) ', g=' num2str(request_pool_size) ', rho=' num2str(rho) ', D_r=' num2str(approx_reservoir_size)])
legend();
yticks(0:0.5:2.5); ylim([0 1.5]);
xlabel('lyapunov time'); ylabel('Root Mean Squared Error');
max_time = max(times(1,:));
% xticks(0:floor(max_time/5/10)*10:max_time); 
axis tight;

