data_dir = './';
L = 22; N = 840; max_lyapunov = 0.0479; dt = 1/4;
% request_pool_size_list = [6 8];
request_pool_size_list = 6;

% del_locality_list = 1;
del_locality_list = 5;
% del_locality = 10;
% del_locality_list = [1 5];
% del_locality_list = [1 5 10];

% beta1 = 0.9; beta2 = 0.999;
beta1 = 0.7; beta2 = 0.9999;
reservoir_fix = true;
% reservoir_fix = false;
if reservoir_fix
    fix_ = 'fix';
else
    fix_ = 'nonfix';
end

% train_steps_list = 1e4;
% train_steps = 2e4 + 1;
% train_steps = 4e4;
% train_steps = 8e4;
train_steps_list = [1e4 2e4+1 4e4 8e4];
% train_steps_list = [5e3 1e4 2e4+1 4e4 8e4];
% train_steps_list = [1e4 2e4+1 4e4];

cost_function = 'pred_time';
% cost_function = 'train_error';
% cost_function = 'pred_time';
% jobid_list = 1:10;
jobid_list = 1:5;
% jobid_list = 6:10;
% jobid_list = 1:4;
% request_pool_size_list = 2:8; del_locality_list = 1; cost_function = 'train_error';
% request_pool_size_list = [2:8]; del_locality_list = 5; cost_function = 'pred_time';
max_iter = 30;
locality_jobid_g = zeros(length(jobid_list), length(request_pool_size_list));

for request_pool_size_id = 1:length(request_pool_size_list)
    request_pool_size = request_pool_size_list(request_pool_size_id);
    Dr = 5040 / request_pool_size;
    final_costs = zeros(length(train_steps_list), length(del_locality_list), length(jobid_list));
    final_localities = zeros(length(train_steps_list), length(del_locality_list), length(jobid_list));
    runtimes = zeros(length(train_steps_list), length(del_locality_list), length(jobid_list));
    for train_steps_index = 1:length(train_steps_list)
        train_steps = train_steps_list(train_steps_index);
        for del_locality_index = 1:length(del_locality_list)
            del_locality = del_locality_list(del_locality_index);
            trajectory_locality = zeros(length(jobid_list), max_iter);
            trajectory_cost = zeros(length(jobid_list), max_iter);
            for jobid_index = 1:length(jobid_list)
                jobid = jobid_list(jobid_index);
                filename = ['KS_GD/result_' cost_function '_del' num2str(del_locality) '_' fix_ '_trainstep' num2str(train_steps) '_node' num2str(Dr) '-L22-N840-radius0.6-numlabs' num2str(request_pool_size) '-jobid' num2str(jobid) '-index_iter1' '_beta1_' num2str(beta1) '_beta2_' num2str(beta2) '.mat'];
                load(filename);
                trajectory_locality(jobid_index, :) = locality_iters;
                trajectory_cost(jobid_index, :) = mean(cost_iters, 1);
                runtimes(train_steps_index, del_locality_index, jobid_index) = runtime;
                final_costs(train_steps_index, del_locality_index, jobid_index) = mean(cost_iters(:,max_iter), 1);
                final_localities(train_steps_index, del_locality_index, jobid_index) = locality_iters(max_iter);
                locality_jobid_g(jobid_index, request_pool_size_id) = locality_iters(max_iter); 
                % locality_jobid_g(jobid_index, request_pool_size_id) = locality_iters(1); 
            end
            % figure(); plot(trajectory_locality.');
            % figure(); plot(trajectory_cost.');
            
            % loc_fig = figure(); plot(L/N*trajectory_locality.');
            % xlabel('iteration'); ylabel('locality'); fontsize(16, 'points'); grid on;
            % fig_name_loc = [data_dir '/', 'KS_GD', '/', 'trajectory_locality_' cost_function '_del' num2str(del_locality) '_' fix_ '_' 'trainstep' num2str(train_steps)];
            % saveas(loc_fig, [fig_name_loc '.jpg']);
            % saveas(loc_fig, [fig_name_loc '.fig']);
    
            % switch cost_function
            %     case 'pred_time'
            %         cost_fig = figure(); plot(-max_lyapunov*dt*trajectory_cost.');
            %         xlabel('iteration'); ylabel('short-term prediction time (\Lambda_1 t)'); fontsize(16, 'points'); grid on;
            %     case 'train_error'
            %         cost_fig = figure(); plot(trajectory_cost.');
            %         xlabel('iteration'); ylabel('train error'); fontsize(16, 'points'); grid on;
            % end
            % fig_name_cost = [data_dir '/', 'KS_GD', '/', 'trajectory_cost_' cost_function '_del' num2str(del_locality) '_' fix_ '_trainstep' num2str(train_steps)];
            % saveas(cost_fig, [fig_name_cost '.jpg']);
            % saveas(cost_fig, [fig_name_cost '.fig']);
    
        end
    end
end
if length(request_pool_size_list) == 1
    for del_locality_index = 1:length(del_locality_list)
        del_locality = del_locality_list(del_locality_index);
        switch cost_function
            case 'pred_time'
                cost_fig = figure(); scatter(train_steps_list, -max_lyapunov*dt*squeeze(final_costs(:, del_locality_index,:)));
                xlabel('N_{batch}'); ylabel('short-term prediction time(\Lambda_1 t)'); 
                xticks(train_steps_list); title(['\Delta n_l=' num2str(del_locality)]);
                % ylim([0.3 1.2]);
                fontsize(16, 'points'); %grid on;
            case 'train_error'
                % cost_fig = figure(); scatter(train_steps_list, squeeze(final_costs(:, del_locality_index,:)));
                cost_fig = figure(); scatter(train_steps_list, squeeze(final_localities(:, del_locality_index,:)));
                xlabel('N_{batch}'); ylabel('train error'); 
                xticks(train_steps_list); title(['\Delta n_l=' num2str(del_locality)]);
                % ylim([0.3 1.2]);
                fontsize(16, 'points'); %grid on;
                % cost_fig = figure(); plot(trajectory_cost.');
                % xlabel('iteration'); ylabel('train error'); fontsize(16, 'points'); grid on;
        end
    end
    
    for train_steps_index = 1:length(train_steps_list)
        train_steps = train_steps_list(train_steps_index);
        switch cost_function
            case 'pred_time'
                cost_fig = figure(); scatter(del_locality_list, -max_lyapunov*dt*squeeze(final_costs(train_steps_index,:,:)));
                xlabel('\Delta n_l'); ylabel('short-term prediction time(\Lambda_1 t)'); 
                xticks(del_locality_list); title(['N_{batch}=' num2str(train_steps)])
                fontsize(16, 'points'); %grid on;
            case 'train_error'
                cost_fig = figure(); plot(trajectory_cost.');
                xlabel('iteration'); ylabel('train error'); fontsize(16, 'points'); grid on;
        end
    end
    
    if true
        runtimes = squeeze(runtimes);
        figure(); errorbar(train_steps_list, mean(runtimes, 2), std(runtimes, [], 2));
        set(gca,'Xscale','log')
        fontsize(16,  'points');
        xlabel('N_{batch}');
        ylabel('Calculation Time (sec)');
        ylabel('calculation time (sec)');
        xticks(train_steps_list);
        grid on;
        switch cost_function
            case 'pred_time'
                filename_runtime = ['-djpeg','chapter3/CalculationTime_GD_PredictionTime.jpg'];
            case 'train_error'
                filename_runtime = ['-djpeg','chapter3/CalculationTime_GD_TrainError.jpg'];
        end
        print(gcf, filename_runtime,'-r600')
    end
else
    figure(); scatter(request_pool_size_list, L/N*locality_jobid_g);
    xlabel('number of reservoirs'); ylabel('locality'); fontsize(16, 'points');
    grid on;
    switch cost_function
            % case 'pred_time'
            %     cost_fig = figure(); scatter(train_steps_list, -max_lyapunov*dt*squeeze(final_costs(:, del_locality_index,:)));
            %     xlabel('N_{batch}'); ylabel('short-term prediction time(\Lambda_1 t)'); 
            %     xticks(train_steps_list); title(['\Delta n_l=' num2str(del_locality)]);
            %     % ylim([0.3 1.2]);
            %     fontsize(16, 'points'); %grid on;
            % case 'train_error'
            %     % cost_fig = figure(); scatter(train_steps_list, squeeze(final_costs(:, del_locality_index,:)));
            %     cost_fig = figure(); scatter(train_steps_list, squeeze(final_localities(:, del_locality_index,:)));
            %     xlabel('N_{batch}'); ylabel('train error'); 
            %     xticks(train_steps_list); title(['\Delta n_l=' num2str(del_locality)]);
            %     % ylim([0.3 1.2]);
            %     fontsize(16, 'points'); %grid on;
            %     % cost_fig = figure(); plot(trajectory_cost.');
            %     % xlabel('iteration'); ylabel('train error'); fontsize(16, 'points'); grid on;
    end
end


final_locality_g = median(locality_jobid_g);
switch cost_function
    case 'train_error'
        final_locality_filename = ['final_localities_TrainError_del1.mat'];
    case 'pred_time'
        final_locality_filename = ['final_localities_PredictionTime_del5.mat'];
end
save(final_locality_filename, 'final_locality_g', 'request_pool_size_list', 'del_locality_list', 'cost_function', 'max_iter');


data_kind = 'KS'; learn = 'LSM_GD_short_prediction_time'; reservoir_kind = 'uniform';
beta = 1e-2; L = 22;  N = 840; jobid = 1662;
errors = zeros(length(request_pool_size_list), 2499);
for res_id = 1:length(request_pool_size_list)
    locality = final_locality_g(res_id);
    num_workers = request_pool_size_list(res_id);
    Dr = 5040 / num_workers; 
    train_steps_test = 8e4;
    
    filename = ['\\nas08c093\data\otsuki\parallelized-reservoir-computing\KSParallelReservoir\' data_kind '\' data_kind '_result_' learn '_' reservoir_kind '_reservoir_' 'beta' num2str(beta) '_train' num2str(train_steps_test) '_node' num2str(Dr) '-L' num2str(L) '-N' num2str(N) '-radius0.6-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter1.mat'];
    load(filename, 'error', 'RMSE_mean', 'lambda_list', 'deltas','resparams', 'delta1_list');
    for l = 1:num_pred % length(pred_marker_array)
        if false
            error_one = log10(error(1, (l-1)*pred_length1+1:(l-1)*pred_length1+pred_length));
            errors(res_id, :) = errors(res_id, :) + error_one;
        else
            errors(res_id, :) = errors(res_id, :) + error(1, (l-1)*pred_length+1:(l-1)*pred_length+pred_length);
        end
        error_pred = error(1, (l-1)*pred_length1+1:(l-1)*pred_length1+pred_length);
        pred_time = find(error_pred>threshold);
        pred_time = pred_time(1);
        % pred_times_each_l(l) = pred_time;
        % for ttt = 1:10:101
        %     sum_E(m, k) = sum_E(m, k) + log(error_pred(ttt))/(ttt*dt);
        % end
        % T_list(h, k) = pred_time(1);
        % % % T_trials(m, l) = pred_time;
        % % % T_error_list2(m) = T_error_list2(m) + (pred_time-1)*dt; %%% dt kakeru???
        % % % ln_error_list2(m) = ln_error_list2(m) + log(error_pred(pred_time)/error_pred(1));
        % % % 
        % % % if pred_time > T_th_max
        % % %     ln_error_list(m) = ln_error_list(m) + log(error_pred(T_th_max)/error_pred(1));
        % % %     T_error_list(m) = T_error_list(m) + (T_th_max-1)*dt;
        % % % else
        % % %     ln_error_list(m) = ln_error_list(m) + log(error_pred(pred_time)/error_pred(1));
        % % %     T_error_list(m) = T_error_list(m) + (pred_time-1)*dt;
        % % % end
        % % % pred_error_locality_list(k, m) = pred_error_locality_list(k, m) + error_pred(T_th_max);
    end
    errors(res_id, :) = errors(res_id, :) / num_pred;
end

% figure(); 
% hold on;
threshold = 1.0;
times = [1:size(errors, 2)]*dt*max_lyapunov;
PredTime = zeros(1, length(request_pool_size_list));
for res_id = 1:length(request_pool_size_list)
    predtime = find(errors(res_id, :)>threshold);
    PredTime(res_id) = predtime(1);
    % legend_ = ['']
    % plot(times, errors(res_id, :), 'DisplayName', legend_);
end
% hold off;
% legend show;
switch cost_function
    case 'pred_time'
        save('GDresult_PredTime_PredTime.mat', 'PredTime');
    case 'train_error'
        save('GDresult_PredTime_TrainError.mat', 'PredTime')
end
PredTime_TE = load('GDresult_PredTime_TrainError.mat', 'PredTime').PredTime;
PredTime_PT = load('GDresult_PredTime_PredTime.mat', 'PredTime').PredTime;

