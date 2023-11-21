
%% errors 
% Ntotal = 1680;
% Ntotal = 5040;
% locality = 50;
% num_workers_list = 2:8;
% L = 22; Ntotal = 5040; locality_list = 1:1:15; num_workers_list = [2];
% L = 22; Ntotal = 2520; locality = 100; num_workers_list = [3 5 6 7 8];
% L = 22; Ntotal = 5040; locality = 100; num_workers_list = [1 2 3 4 6 8];
% L = 26; Ntotal = 3360; locality = 100; num_workers_list = [2 6 8];
% L = 26; Ntotal = 5880; locality = 100; num_workers_list = [1 2 4 6 8];
% L = 44; Ntotal = 5040; locality_list = [60 55 50 45 40]; num_workers_list = [6]; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% L = 44; Ntotal = 5040; locality_list = [60 55 50 45 40 35 30]; num_workers_list = [8 10]; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
L = 44; Ntotal = 5040; locality_list = [55 50 45 40 35 30]; num_workers_list = [12]; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% L = 50; Ntotal = 5040; locality = 50; num_workers_list = [2 4 5 6 7 8 10 12 14];
% L = 52; Ntotal = 80016; locality = 9; num_workers_list = [12];
% L = 52; Ntotal = 80016; locality = 6; num_workers_list = [16];
% L = 66; Ntotal = 15120; locality = 50; num_workers_list = [3 6 12];
% L = 88; Ntotal = 5040; locality_list = [15 50 70]; num_workers_list = 15;
% jobid_list = 1;
% jobid_list = [2 3 4 6];
% jobid_list = 2:5;
jobid_list = 1:5;
% jobid_list = 1:10;
% num_workers_list = [2 3 4 8];
% num_workers_list = [1 2 4 6];
% num_workers_list = [6 8 10 12 14];
% num_workers_list = [1 2 4 6 8 10 12];
% num_workers_list = [1 2 3 4 6 8];
% num_workers_list = [2 3 4 5 6 7 8 10 12 14];
% num_workers_list = [5 6 7 8];
% num_workers_list = [8 10 12 14 15];
% num_workers_list = [2 3 4 5 6 7 8];
% num_workers_list = [2 4 8 12 15];
% pred_length = 2499; num_pred = 49;
pred_length = 2499; num_pred = 20;
lineWidth = 1.5;
dt = 1/4; max_lyapunov = 0.0743;
n_steps = size(trajectories_true, 2);
n_data = size(trajectories_true, 1);
times = repmat(0:dt*max_lyapunov:(n_steps-1)*dt*max_lyapunov, n_data, 1);
locations = repmat((1:n_data).', 1, n_steps);
threshold = 0.30;
for num_workers = num_workers_list
T_list = zeros(1, length(locality_list));
figure();
for k = 1:length(locality_list)
locality = locality_list(k);
errors = zeros(length(locality_list), pred_length);
for jobid = jobid_list
    Dr = Ntotal / num_workers;
    filename = ['\\nas08c093\data\otsuki\parallelized-reservoir-computing\KSParallelReservoir\KS\KS_result_LSM_common_uniform_train80000_node' num2str(Dr) '-L' num2str(L) '-radius0.6-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter1.mat'];
    load(filename, 'error')
    
    % pred_length = 2899; num_pred = 49;
    % figure();
    % num_pred = 49;
    for l = 1:num_pred % length(pred_marker_array)
        errors(k, :) = errors(k, :) + error(1, (l-1)*pred_length+1:(l-1)*pred_length+pred_length);
    end
end
    errors(k, :) = errors(k, :) / (num_pred*length(jobid_list));
    
    hold on;
    % plot(error);
    % plot(times(1,1:pred_length), errors(1, 1:pred_length), 'DisplayName', ['locality=' num2str(locality)]); 
    plot(times(1,1:pred_length), errors(k, 1:pred_length), 'LineWidth', lineWidth, 'DisplayName', ['l=' num2str(locality)]); 
    % plot(times(1,:), error(1, 1:n_steps), 'DisplayName', ['train steps=' num2str(train_steps)]); 
    hold off;

    pred_time = find(errors(k, :)>threshold);
    T_list(k) = pred_time(1);
end
% sgtitle(['L=' num2str(L) ', g=' num2str(request_pool_size) ', rho=' num2str(rho) ', D_r=' num2str(approx_reservoir_size)])
sgtitle(['RMSE L=' num2str(L) ', num reservoirs: ' num2str(num_workers) ', Ntotal=' num2str(Ntotal)]);
legend(); fontsize(16, 'points');
% yticks(0:0.5:2.5); ylim([0 1.5]);
xlabel('lyapunov time'); ylabel('Root Mean Squared Error');
max_time = max(times(1,:));
% xticks(0:floor(max_time/5/10)*10:max_time); 
axis tight; grid on;
% xlim([0 6]);

% show prediction time
figure(); plot(locality_list, T_list);
xlabel('locality'); ylabel('short-term prediction time');
legend(); fontsize(16, 'points'); axis tight; grid on;
end
%}
