%%
% 一つのリザバーのノードが一定
%% errors 
% Ntotal = 1680;
% Ntotal = 5040;
% locality = 50;
% num_workers_list = 2:8;

% L = 22; Ntotal = 5040; N=840; locality_list = [7 8 9 10 11 12 20]; num_workers_list = [6]; % jobid_list = 2:3; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% L = 22; Ntotal = 5880; N=840; locality_list = [9 10 11 12]; num_workers_list = [6];  jobid_list = 6:10; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% L = 22; Ntotal = 5040; N=840; locality_list = 40:130; num_workers_list = [6];  jobid_list = 1:4; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
L = 22; Ntotal = 5040; N=840; locality_list = 40:130; num_workers_list = [6];  jobid_list = 1:4; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];

%% 
% L = 22; Ntotal = 5040; N=840; locality_list = 20:20:280; num_workers_list = [2 3 4 5 6 7 8]; jobid_list = 1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
%%
% data_kind = 'KS_slow'; L = 44; dt = 1/8; num_workers_list = 2*[2 3 4 5 6 7 8]; Ntotal_list = 5040 * num_workers_list; N=1680; train_steps = 160000; locality_list = 20:20:280; jobid_list = 1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];

%% フィードバックありの予測誤差
% data_kind = 'KS_slow'; L = 22; dt = 1/8; Ntotal_list = 5040 * [2:8]; N=840; train_steps = 160000; locality_list = 20:20:280; num_workers_list = [2 3 4 5 6 7 8]; jobid_list = 1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% data_kind = 'KS_slow_short'; L = 22; dt = 1/8; Ntotal_list = 5040 * [2:8]; N=840; train_steps = 80000; locality_list = 20:20:280; num_workers_list = [2 3 4 5 6 7 8]; jobid_list = 1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% data_kind = 'KS'; L = 22; dt = 1/4; num_workers_list = [2 3 4 5 6 7 8]; Ntotal_list = 5040 * num_workers_list; N=840; train_steps = 80000; locality_list = 0:20:280; jobid_list = 1:2; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% data_kind = 'KS'; L = 44; dt = 1/4; num_workers_list = 2*[2 3 4 5 6 7 8]; Ntotal_list = 5040 * num_workers_list; N=1680; train_steps = 80000; locality_list = 0:20:280; jobid_list = 1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];

% data_kind = 'KS'; L = 22; dt = 1/4; num_workers_list = [2 3 4 5 6 7 8]; Ntotal_list = 5040 * num_workers_list; N=840; train_steps = 80000; locality_list = 60:20:180; jobid_list = 1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
data_kind = 'KS'; L = 22; dt = 1/4; num_workers_list = [2]; Ntotal_list = 5040 * num_workers_list; N=840; train_steps = 80000; locality_list = 40:130; jobid_list = 1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];

%% フィードバックなしの予測誤差
% data_kind = 'KS'; L = 22; dt = 1/4; num_workers_list = [7 8]; Ntotal_list = 5040 * num_workers_list; N=840; train_steps = 80000; locality_list = 60:20:120; jobid_list = 100; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];

%%
% L = 44; Ntotal = 10080; N=1680; locality_list = 20:20:160; num_workers_list = 4:2:16; jobid_list = 1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% L = 44; num_workers_list = 4:2:16; Ntotal_list = 5040 * num_workers_list; N=1680; locality_list = 20:20:160; jobid_list = 1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% L = 44; num_workers_list = 4:2:16; Ntotal_list = 5040 * num_workers_list; N=1680; locality_list = 180:20:280; jobid_list = 1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];

% % L = 22; Ntotal = 15120; locality_list = 20:20:160; num_workers_list = [2 3 4 5 6 7 8]; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% L = 22; Ntotal = 5040; locality_list = 20:20:160; num_workers_list = [2 3 4 5 6 7 8 10]; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% L = 44; Ntotal = 5040; locality_list = 10:10:80; num_workers_list = [2 3 4 5 6 7 8 10 12 14 15]; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% % L = 66; Ntotal = 5040; locality_list = 10:10:80; num_workers_list = [1 2 3 4 5 6 7 8]; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% % L = 66; Ntotal = 5040; locality_list = 10:10:80; num_workers_list = [10 12 14 15]; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% L = 66; Ntotal = 5040; locality_list = 10:10:80; num_workers_list = [1:8 10 12 14 15]; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];

% L = 22; Ntotal = 5040; locality_list = 1:1:15; num_workers_list = [2];
% L = 22; Ntotal = 2520; locality = 100; num_workers_list = [3 5 6 7 8];
% L = 22; Ntotal = 5040; locality = 100; num_workers_list = [1 2 3 4 6 8];
% L = 26; Ntotal = 3360; locality = 100; num_workers_list = [2 6 8];
% L = 26; Ntotal = 5880; locality = 100; num_workers_list = [1 2 4 6 8];
% L = 44; Ntotal = 5040; locality_list = [60 55 50 45 40]; num_workers_list = [6]; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% L = 44; Ntotal = 5040; locality_list = [60 55 50 45 40 35 30]; num_workers_list = [8 10]; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% L = 44; Ntotal = 5040; locality_list = [55 50 45 40 35 30]; num_workers_list = [12]; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% L = 50; Ntotal = 5040; locality = 50; num_workers_list = [2 4 5 6 7 8 10 12 14];
% L = 52; Ntotal = 80016; locality = 9; num_workers_list = [12];
% L = 52; Ntotal = 80016; locality = 6; num_workers_list = [16];
% L = 66; Ntotal = 15120; locality = 50; num_workers_list = [3 6 12];
% L = 88; Ntotal = 5040; locality_list = [15 50 70]; num_workers_list = 15;
% jobid_list = 1;
% jobid_list = 2:3;
% jobid_list = [2 3 4 6];
% jobid_list = 2:5;
% jobid_list = 1:5;
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
% dt = 1/4;
% max_lyapunov = 0.0743;
switch L
    case 22
        % max_lyapunov = 0.0825;
        max_lyapunov = 0.0479;
    case 44
        max_lyapunov = 0.09;
end
% n_steps = size(trajectories_true, 2);
n_steps = 49980;
show_error_plot = true;
% show_error_plot = false
% n_data = size(trajectories_true, 1);
n_data = N;
times = repmat(0:dt*max_lyapunov:(n_steps-1)*dt*max_lyapunov, n_data, 1);
locations = repmat((1:n_data).', 1, n_steps);
threshold = 0.30;
T_list = zeros(length(num_workers_list), length(locality_list));
RMSE_list = zeros(length(num_workers_list), length(locality_list));
% RMSE_pred = true;
RMSE_pred = false;
if RMSE_pred
    RMSE_pred_list = zeros(length(num_workers_list), length(locality_list));
end
for h = 1:length(num_workers_list)
    num_workers = num_workers_list(h);
    Ntotal = Ntotal_list(h);
    train_errors = zeros(length(jobid_list), length(locality_list));
    if show_error_plot
        figure();
    end
    for k = 1:length(locality_list)
        locality = locality_list(k);
        errors = zeros(length(locality_list), pred_length);
        % T_trials = zeros(length(jobid_list), length(locality_list));
        T_trials = zeros(length(jobid_list), num_pred);
        for m = 1:length(jobid_list)
            jobid = jobid_list(m);
            Dr = Ntotal / num_workers;
            % filename = ['\\nas08c093\data\otsuki\parallelized-reservoir-computing\KSParallelReservoir\KS\KS_result_LSM_common_uniform_train80000_node' num2str(Dr) '-L' num2str(L) '-radius0.6-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter1.mat'];
            % filename = ['\\nas08c093\data\otsuki\parallelized-reservoir-computing\KSParallelReservoir\KS\KS_result_LSM_common_uniform_reservoir_train80000_node' num2str(Dr) '-L' num2str(L) '-N' num2str(N) '-radius0.6-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter1.mat'];
            filename = ['\\nas08c093\data\otsuki\parallelized-reservoir-computing\KSParallelReservoir\' data_kind '\' data_kind '_result_LSM_common_uniform_reservoir_train' num2str(train_steps) '_node' num2str(Dr) '-L' num2str(L) '-N' num2str(N) '-radius0.6-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter1.mat'];
            if RMSE_pred
                load(filename, 'error', 'RMSE_mean', 'RMSE_pred_data');
            else
                load(filename, 'error', 'RMSE_mean');
            end
            train_errors(m, k) =  RMSE_mean;
            
            if RMSE_pred
                RMSE_pred_list(h, k) = mean(RMSE_pred_data);
            else
                % % pred_length = 2899; num_pred = 49;
                % % figure();
                % % num_pred = 49;
                for l = 1:num_pred % length(pred_marker_array)
                    error_pred = error(1, (l-1)*pred_length+1:(l-1)*pred_length+pred_length);
                    % errors(k, :) = errors(k, :) + error(1, (l-1)*pred_length+1:(l-1)*pred_length+pred_length);
                    errors(k, :) = errors(k, :) + error_pred;
                    pred_time = find(error_pred>threshold);
                    if size(pred_time, 2) == 0
                        T_trials(m, l) = 0;
                    else
                        % T_list(h, k) = pred_time(1);
                        T_trials(m, l) = pred_time(1);
                    end
                end
            end
        end
        errors(k, :) = errors(k, :) / (num_pred*length(jobid_list));
        % figure();
        % errorbar(mean(T_trials, 2), std(T_trials, 0, 2));
        % xlabel('jobid'); ylabel('short prediction time');
        % title(['locality: ' num2str(locality)]);
        if show_error_plot
            hold on;
            % plot(error);
            % plot(times(1,1:pred_length), errors(1, 1:pred_length), 'DisplayName', ['locality=' num2str(locality)]); 
            plot(times(1,1:pred_length), errors(k, 1:pred_length), 'LineWidth', lineWidth, 'DisplayName', ['l=' num2str(locality)]); 
            % plot(times(1,:), error(1, 1:n_steps), 'DisplayName', ['train steps=' num2str(train_steps)]); 
            hold off;
        end
        
        % pred_time = find(errors(k, :)>threshold);
        % T_list(h, k) = pred_time(1);
        pred_time = median(median(T_trials));
        % pred_time = mean(mean(T_trials));
        T_list(h, k) = pred_time;
        RMSE_list(h, :) = mean(train_errors, 1).';
    end
    if show_error_plot
        % sgtitle(['L=' num2str(L) ', g=' num2str(request_pool_size) ', rho=' num2str(rho) ', D_r=' num2str(approx_reservoir_size)])
        sgtitle(['RMSE L=' num2str(L) ', num reservoirs: ' num2str(num_workers) ', Ntotal=' num2str(Ntotal)]);
        legend(); fontsize(16, 'points');
        xlabel('lyapunov time'); ylabel('Root Mean Squared Error');
        % max_time = max(times(1,:));
        max_time = 10;
        xticks(0:floor(max_time/5/10)*10:max_time); xlim([0 max_time]);
        yticks(0:0.5:2.5); ylim([0 1.5]);
        axis tight; grid on; legend('Location', 'eastoutside');
    end

    
    % % show prediction time
    % figure(); plot(locality_list, T_list(h, :));
    % sgtitle(['L=' num2str(L) ', num reservoirs: ' num2str(num_workers) ', Ntotal=' num2str(Ntotal)]);
    % xlabel('locality'); ylabel('short-term prediction time');
    % legend(); fontsize(16, 'points'); axis tight; grid on;
end
%}

figure(); 
for h = 1:length(num_workers_list)
    hold on;
    num_workers = num_workers_list(h);
    plot(locality_list/n_data*L, T_list(h, :), 'DisplayName', ['g=' num2str(num_workers)]);
    hold off;
end
% sgtitle(['L=' num2str(L) ', Ntotal=' num2str(Ntotal)]);
xlabel('locality (space)'); %xlabel('locality'); 
ylabel('short-term prediction time');
legend('Location', 'eastoutside'); fontsize(16, 'points'); axis tight; grid on;

% figure(); surf(repmat(num_workers_list.', 1, length(locality_list)), repmat(locality_list, length(num_workers_list), 1), T_list);
% xlabel('number of reservoirs'); ylabel('locality'); title([data_kind ' L=' num2str(L)]);
% colorbar; view(0, 90); fontsize(16, 'points'); axis tight;

figure(); surf(repmat(num_workers_list.', 1, length(locality_list)), L/n_data*repmat(locality_list, length(num_workers_list), 1), max_lyapunov*dt*T_list, 'FaceColor', 'interp');
xlabel('number of reservoirs'); ylabel('locality'); title(['short-term prediction time: ' data_kind ' L=' num2str(L) 'fix Dr']);
colorbar; view(0, 90); fontsize(16, 'points'); axis tight;

figure(); surf(repmat(num_workers_list.', 1, length(locality_list)), L/n_data*repmat(locality_list, length(num_workers_list), 1), RMSE_list, 'FaceColor', 'interp');
xlabel('number of reservoirs'); ylabel('locality'); title(['RMSE ' data_kind 'L=' num2str(L) ', fix Dr']);
colorbar; view(0, 90); fontsize(16, 'points'); axis tight;

% figure(); surf(repmat(num_workers_list.', 1, length(locality_list)), repmat(locality_list, length(num_workers_list), 1), T_list, 'FaceColor', 'interp', 'EdgeColor', 'interp');
% xlabel('number of reservoirs'); ylabel('locality'); title([data_kind 'L=' num2str(L)]);
% colorbar; view(0, 90); fontsize(16, 'points'); axis tight;