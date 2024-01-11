data_dir = './';
L = 22; N = 840; max_lyapunov = 0.0479; dt = 1/4;

% beta1 = 0.9; beta2 = 0.999;
beta1 = 0.7; beta2 = 0.9999;
reservoir_fix = true;
% reservoir_fix = false;
if reservoir_fix
    fix_ = 'fix';
else
    fix_ = 'nonfix';
end


train_steps = 5e3;
train_steps = 1e4;
train_steps = 2e4 + 1;
% train_steps = 4e4;
% train_steps = 8e4;

del_locality = 1;
% del_locality = 5;
% del_locality = 10;
% del_locality = 15;

cost_function = 'pred_time';
cost_function = 'train_error';
% cost_function = 'pred_time';
% jobid_list = 1:10;
jobid_list = 1:5;
% jobid_list = 6:10;
% jobid_list = [1 6:10];
% jobid_list = 6:8;
% jobid_list = 1e5 + [11:15];
% jobid_list = 1:4;
max_iter = 30;
trajectory_locality = zeros(length(jobid_list), max_iter);
trajectory_cost = zeros(length(jobid_list), max_iter);
for jobid_index = 1:length(jobid_list)
    jobid = jobid_list(jobid_index);
    filename = ['KS_GD/result_' cost_function '_del' num2str(del_locality) '_' fix_ '_trainstep' num2str(train_steps) '_node840-L22-N840-radius0.6-numlabs6-jobid' num2str(jobid) '-index_iter1' '_beta1_' num2str(beta1) '_beta2_' num2str(beta2) '.mat'];
    load(filename);
    trajectory_locality(jobid_index, :) = locality_iters;
    trajectory_cost(jobid_index, :) = mean(cost_iters, 1);
end

% figure(); plot(trajectory_locality.');
% figure(); plot(trajectory_cost.');

loc_fig = figure(); plot(L/N*trajectory_locality.');
xlabel('iteration'); ylabel('locality'); fontsize(16, 'points'); grid on;
fig_name_loc = [data_dir '/', 'KS_GD', '/', 'trajectory_locality_' cost_function '_del' num2str(del_locality) '_' fix_ '_' 'trainstep' num2str(train_steps)];
saveas(loc_fig, [fig_name_loc '.jpg']);
saveas(loc_fig, [fig_name_loc '.fig']);
switch cost_function
    case 'pred_time'
        cost_fig = figure(); plot(-max_lyapunov*dt*trajectory_cost.');
        xlabel('iteration'); ylabel('short-term prediction time (\Lambda_1 t)'); fontsize(16, 'points'); grid on;
    case 'train_error'
        cost_fig = figure(); plot(trajectory_cost.');
        xlabel('iteration'); ylabel('train error'); fontsize(16, 'points'); grid on;
end

fig_name_cost = [data_dir '/', 'KS_GD', '/', 'trajectory_cost_' cost_function '_del' num2str(del_locality) '_' fix_ '_trainstep' num2str(train_steps)];
saveas(cost_fig, [fig_name_cost '.jpg']);
saveas(cost_fig, [fig_name_cost '.fig']);


% threshold = 1.0;
% times = [1:size(errors, 2)]*dt*max_lyapunov;
% PredTime = zeros(1, length(request_pool_size_list));
% for res_id = 1:length(request_pool_size_list)
%     predtime = find(errors(res_id, :)/num_pred>threshold);
%     PredTime(res_id) = predtime(1);
%     % legend_ = ['']
%     % plot(times, errors(res_id, :), 'DisplayName', legend_);
% end
% 
% save('GDresult_PredTime_TrainError.mat', 'PredTime')
% 
% save('GDresult_PredTime_PredTime.mat', 'PredTime')
% PredTime_TE = load('GDresult_PredTime_TrainError.mat', 'PredTime')



