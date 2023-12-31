
%Author: Jaideep Pathak, PhD candidate at the University of Maryland,
%College Park.
%email: jpathak@umd.edu or jaideeppathak244@gmail.com



% function jobid = parallel_reservoir_benchmarking(request_pool_size)
 data_dir = './';
 % request_pool_size = 1;
 % request_pool_size = 8;
 request_pool_size = 16;
 % index_file = matfile('/lustre/jpathak/KS100/testing_ic_indexes.mat');
 index_file = matfile([data_dir 'testing_ic_indexes.mat']);
 %index_file = matfile('KS100/testing_ic_indexes.mat');

 full_pred_marker_array = index_file.testing_ic_indexes;
 
 num_indices = length(full_pred_marker_array);
 
 num_divided_jobs = 10;
 
 indices_per_job = num_indices/num_divided_jobs;

 % rho_list = 0.5:0.3:1.7;
 % locality_list = 4:2:8;

 % rho_list = 0.8:0.2:1.4;
 % locality_list = 7:8;
 
 % rho_list = 0.8;
 % locality_list = 8;
 
 % rho_list = 1.2:0.05:1.6;
 % rho_list = 0.05:0.05:0.15;
 % rho_list = 0.4:0.2:1.6;
 % rho_list = 0.2:0.1:0.3;
 rho_list = 0.6;
 % rho_list = [1.3];
 % locality_list = 0;
 locality_list = 8;
 % locality_list = 16;
 % train_steps_list = 8e4:2e4:14e4;
 train_steps_list = 8e4;
 % rho_list = 0.2:1:1.7;
 % locality_list = 3:4:8;
 h = waitbar(0,'Please wait...');
 progress = 0;
 for index_iter = 1:1
 for rho = rho_list
 for locality_ = locality_list
 for train_steps = train_steps_list
    tic;
    partial_pred_marker_array = full_pred_marker_array((index_iter-1)*indices_per_job + 1:index_iter*indices_per_job);

    pred_marker_array = Composite(request_pool_size);
    which_index_iter = Composite(request_pool_size);
    rho_array = Composite(request_pool_size);
    locality_array = Composite(request_pool_size);

    % pred_marker_array = zeros(1, request_pool_size);
    % which_index_iter = zeros(1, request_pool_size);
    % rho_array = zeros(1, request_pool_size);
    % locality_array = zeros(1, request_pool_size);

    % pred_marker_array = partial_pred_marker_array;
    % which_index_iter = index_iter;
    % rho_array = rho;
    % locality_array = locality_;
    % 
    if true
        for i=1:length(pred_marker_array)
            pred_marker_array{i} = partial_pred_marker_array;
            which_index_iter{i} = index_iter;
            rho_array{i} = rho;
            locality_array{i} = locality_;
        end
    end

    request_pool_size
    spmd(request_pool_size)
    % for reservoir_id = 2:request_pool_size
        
        jobid = 1;
        
        data_kind = 'KS';
        % data_kind = 'CGL';
        switch data_kind
            case 'CGL'
                L = 44;
                % L = 36;
                % L = 22;
                % L = 8;
                % N = 32;
                N = 64; 
                % N = 128;
                c1 = -2; c2 = 2;
                % train_steps = 80000;
                % train_steps = 100000;
                % train_steps = 120000;
                % train_steps = 140000;
                % train_steps = 200000;
                % train_steps = 300000;
                % train_steps = 700000;
                test_steps = 20000;
                % m = matfile([data_dir 'CGL_L', num2str(L) '_N_', num2str(N) '_dps', num2str(train_steps) '.mat']); % CGL
                tf = matfile([data_dir 'CGL_L', num2str(L) '_N_' num2str(N) '_dps' num2str(test_steps) 'c1_' num2str(c1) 'c2_' num2str(c2) '.mat']); % CGL
            case 'KS'
                L = 22; N = 64; 
                % L = 44; N = 128; 
                train_steps = 80000;
                % train_steps = 300000;
                test_steps = 20000;
                % m = matfile([data_dir 'train_input_sequence.mat']); % KS
                tf = matfile([data_dir 'test_input_sequence_L44.mat']); % KS
                % tf = matfile([data_dir 'test_input_sequence.mat']); % KS
        end
        % fprintf(['use ', data_kind, '\n']);
        % m = matfile('/lustre/jpathak/KS100/train_input_sequence.mat'); 
%        m = matfile('KS100/train_input_sequence.mat');
%        tf = matfile('KS100/test_input_sequence.mat');
        % tf = matfile('/lustre/jpathak/KS100/test_input_sequence.mat');
        
        % sigma = 0.5;  %% simple scaling of data by a scalar
        sigma = 0.5;  %% simple scaling of data by a scalar
        
        % train_uu = m.train_input_sequence;
        % [len, num_inputs] = size(m, 'train_input_sequence');
        % clear m;
        % m = [];
        % [len, num_inputs] = size(train_uu);

        test_uu = tf.test_input_sequence;
        % clear tf;
        tf = [];
        [len, num_inputs] = size(test_uu);

        % num_workers = numlabs; %numlabs is a matlab func that returns the number of workers allocated. equal to request_pool_size\
        num_workers = request_pool_size;

        % chunk_size = num_inputs/numlabs; %%%%%%%%%% MUST DIVIDE (each reservoir responsible for this chunk)
        chunk_size = num_inputs/num_workers; %%%%%%%%%% MUST DIVIDE (each reservoir responsible for this chunk)
        % fprintf('chunk size: %f\n', chunk_size);
        % fprintf('num_inputs: %f, numlabs: %f\n', num_inputs, numlabs);
        l = labindex; % labindex is a matlab function that returns the worker index
        % l = reservoir_id;

        chunk_begin = chunk_size*(l-1)+1;

        chunk_end = chunk_size*l;

        % locality = 6; % there are restrictions on the allowed range of this parameter. check documentation
        locality = locality_array; % there are restrictions on the allowed range of this parameter. check documentation
   
        rear_overlap = indexing_function_rear(chunk_begin, locality, num_inputs, chunk_size);  %spatial overlap on the one side
        
        forward_overlap = indexing_function_forward(chunk_end, locality, num_inputs, chunk_size);  %spatial overlap on the other side
        if size(rear_overlap, 2) ~= locality
            display(chunk_begin);
            fprintf('rear_overlap: %d\n', size(rear_overlap, 2))
        end
        if size(forward_overlap) ~= locality
            display(chunk_end);
            fprintf('forward_overlap: %d\n', size(forward_overlap, 2));
        end
        overlap_size = length(rear_overlap) + length(forward_overlap); 
        
        % approx_reservoir_size = 5000;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        approx_reservoir_size = 7000;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 8000;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 10000;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 12000;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 15000;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)

        avg_degree = 3; %average connection degree

        resparams.sparsity = avg_degree/approx_reservoir_size;
        
        resparams.degree = avg_degree;
        
        nodes_per_input = round(approx_reservoir_size/(chunk_size+overlap_size));

        resparams.N = nodes_per_input*(chunk_size+overlap_size); % exact number of nodes in the network
        
        resparams.discard_length = 1000;  %number of time steps used to discard transient (generously chosen)

        resparams.train_length = train_steps-resparams.discard_length;  %number of time steps used for training
        % resparams.train_length = 79000;  %number of time steps used for training
        % resparams.train_length = 39000;  %number of time steps used for training
        
        % sync_length = 32; % use a short time series to synchronize to the data at the prediction_marker
        sync_length = 1000; % use a short time series to synchronize to the data at the prediction_marker
        
        % resparams.predict_length = 2999;  %number of steps to be predicted
        resparams.predict_length = test_steps-sync_length-1;  %number of steps to be predicted

        % resparams.radius = 0.6; % spectral radius of the reservoir
        resparams.radius = rho; % spectral radius of the reservoir
        
        resparams.beta = 0.0001; % ridge regression regularization parameter
        
        % u = zeros(len, chunk_size + overlap_size); % this will be populated by the input data to the reservoir
        % 
        % u(:,1:locality) = train_uu(1:end, rear_overlap);
        % 
        % u(:,locality+1:locality+chunk_size) = train_uu(1:end, chunk_begin:chunk_end);
        % 
        % u(:,locality+chunk_size+1:2*locality+chunk_size) = train_uu(1:end,forward_overlap);
        % 
        % u = sigma*u;

        test_u = zeros(20000, chunk_size + overlap_size); % this will be populated by the input data to the reservoir

        test_u(:,1:locality) = test_uu(1:end, rear_overlap);

        test_u(:,locality+1:locality+chunk_size) = test_uu(1:end, chunk_begin:chunk_end);

        test_u(:,locality+chunk_size+1:2*locality+chunk_size) = test_uu(1:end,forward_overlap);

        test_u = sigma*test_u;

        %% load trained reservoir
        filename = [data_dir '/', data_kind, '/', data_kind '_reservoir' num2str(l), 'train', num2str(train_steps), '_node', num2str(approx_reservoir_size) '-L' num2str(L) '-radius' num2str(rho) '-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter', num2str(which_index_iter) '.mat'];
        trained_reservoir = load(filename);

        A = trained_reservoir.A; w_in = trained_reservoir.w_in;
        w_out = trained_reservoir.w_out;
        x = zeros(resparams.N, 1);

        pred_collect = res_predict(x, w_out, A, w_in, transpose(test_u), resparams, jobid, locality, chunk_size, pred_marker_array, sync_length);
        fprintf('finished prediction\n');
        % fprintf('start res train predict');
        % [x, w_out, A, w_in, RMSE] = train_reservoir(resparams, u.', l, jobid, locality, chunk_size);

        % [pred_collect, RMSE] = res_train_predict(transpose(u), transpose(test_u), resparams, jobid, locality, chunk_size, pred_marker_array, sync_length);
        % fprintf('pred_collect');
        collated_prediction = gcat(pred_collect,1,1);
        toc;
    end

    % runtime = toc;
    approx_reservoir_size = approx_reservoir_size{1};
    locality = locality{1};
    chunk_size = chunk_size{1};
    num_workers = num_workers{1};
    jobid = jobid{1};
    % data_file = m{1};
    % test_file = tf{1};
    data_kind = data_kind{1};
    Wres = gather(A);
    Win = gather(w_in);
    Wout = gather(w_out);

    L = L{1}; N = N{1}; train_steps = train_steps{1}; test_steps = test_steps{1};
    switch data_kind
        case 'CGL'
            c1 = c1{1}; c2 = c2{1}; 
            dt = 0.07;
            % data_file = load([data_dir 'CGL_L18_N_32_dps80000.mat']); % CGL
            test_file = load([data_dir 'CGL_L' num2str(L) '_N_' num2str(N) '_dps' num2str(test_steps) 'c1_' num2str(c1) 'c2_' num2str(c2) '.mat']); % CGL
        case 'KS'
            dt = 1/4;
            % data_file = load([data_dir 'train_input_sequence.mat']); % KS
            test_file = load([data_dir 'test_input_sequence_L44.mat']); % KS
            % test_file = load([data_dir 'test_input_sequence.mat']); % KS
    end
    resparams = resparams{1};
    sync_length = sync_length{1};
    sigma = sigma{1};
    num_inputs = num_inputs{1};
    pred_collect = collated_prediction{1};
    pred_marker_array = pred_marker_array{1};
    num_preds = length(pred_marker_array);
    diff = zeros(num_inputs, num_preds*resparams.predict_length);
    trajectories_true = zeros(num_inputs, num_preds*resparams.predict_length);
    
    % RMSE_mean = 0;
    % for i = 1:num_workers
    %     RMSE_mean = RMSE_mean + RMSE{i};
    % end
    % RMSE_mean = RMSE_mean / num_workers;

    for pred_iter = 1:num_preds
        prediction_marker = pred_marker_array(pred_iter);
        trajectories_true(:, (pred_iter-1)*resparams.predict_length + 1: pred_iter*resparams.predict_length) = transpose(sigma*test_file.test_input_sequence(prediction_marker+sync_length + 1:prediction_marker+sync_length + resparams.predict_length,:));
        diff(:, (pred_iter-1)*resparams.predict_length+1:pred_iter*resparams.predict_length) ...
            = transpose(sigma*test_file.test_input_sequence(prediction_marker+sync_length +1:prediction_marker + sync_length + resparams.predict_length,:))...
        -  pred_collect(:,(pred_iter-1)*resparams.predict_length+1:pred_iter*resparams.predict_length);
        error = sqrt(mean(diff.^2, 1));
    end

    which_index_iter = which_index_iter{1};
%    filename = ['KS100-' num2str(approx_reservoir_size) '-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter', num2str(which_index_iter)];
    % filename = ['/lustre/jpathak/KS100/KS100-' num2str(approx_reservoir_size) '-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter', num2str(which_index_iter)];
    % filename = [data_dir '/KS100-' num2str(approx_reservoir_size) '-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter', num2str(which_index_iter)];
    
    
    n_steps = size(trajectories_true, 2);
    n_data = size(trajectories_true, 1);
    times = repmat(0:dt:(n_steps-1)*dt, n_data, 1);
    locations = repmat((1:n_data).', 1, n_steps);
    max_value = max(max(trajectories_true)); min_value = min(min(trajectories_true));
    figure(); 
    subplot(3, 1, 1); surf(times, locations, [trajectories_true(1:2:end-1,:); trajectories_true(2:2:end,:)]); view(0, 90); shading interp, axis tight; xlabel('time'); ylabel('location'); title('target'); colorbar; clim([min_value max_value]); xlim([0 50]);
    subplot(3, 1, 2); surf(times, locations, [pred_collect(1:2:end-1,:); pred_collect(2:2:end,:)]); view(0, 90); shading interp, axis tight; xlabel('time'); ylabel('location'); title('output'); colorbar; clim([min_value max_value]); xlim([0 50]);
    subplot(3, 1, 3); surf(times, locations, [diff(1:2:end-1,:); diff(2:2:end, :)]); view(0, 90); shading interp, axis tight; xlabel('time'); ylabel('location'); title('error'); colorbar; clim(2 * [min_value max_value]); xlim([0 50]);
    sgtitle([data_kind ' rho: ' num2str(resparams.radius) ', locality: ' num2str(locality)]);
    
    figure(); plot(times(1,:), sqrt(mean(diff.^2, 1)));
    xlabel('time (lyapunov*second'); ylabel('RMSE'); title('error');
    
    progress = progress + 1;
    total = size(rho_list, 2) * size(locality_list, 2) * request_pool_size;
    h = waitbar(progress/total,h,... 
    sprintf('progress: %d/%d', progress, total));

    filename = [data_dir '/', data_kind, '/', data_kind, 'result_train', num2str(train_steps), '_node', num2str(approx_reservoir_size) '-L' num2str(L) '-radius' num2str(rho) '-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter', num2str(which_index_iter) '.mat'];
    % filename = [data_dir '/', data_kind, '/', data_kind '_reservoir' num2str(l), 'train', num2str(train_steps), '_node', num2str(approx_reservoir_size) '-L' num2str(L) '-radius' num2str(rho) '-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter', num2str(which_index_iter) '.mat'];
    % save(filename, 'pred_collect', 'error', 'diff', 'resparams', 'RMSE_mean', 'pred_marker_array', 'trajectories_true');
    % save(filename, 'A', 'w_in', 'w_out', 'l', 'resparams', 'RMSE', 'locality', 'chunk_size');
    % dsave filename pred_collect A w_in w_out l resparams RMSE locality chunk_size;
    save(filename, 'Wres', 'Win', 'Wout', 'resparams', 'trajectories_true', 'pred_collect', 'diff', 'error', 'pred_marker_array', 'locality', 'chunk_size');
    display(filename);
 end
 end
 end
 end
 close(h);
 
% train_steps_list = [];
% train_steps_list = 8e4:2e4:14e4;
% jobid = jobid{1};
% L = L{1}; resparams = resparams{1}; data_kind = data_kind{1}; chunk_size = chunk_size{1};
% which_index_iter = which_index_iter{1}; N = N{1}; locality_array = locality_array{1};


train_steps_list = [2e4+1 4e4:2e4:14e4];
n_grids = size(train_steps_list, 2);
rho = 1.6;

n_steps = 18999; approx_reservoir_size = 7000; locality = 8; num_workers = 8;
n_data = 64; dt = 0.07; times = repmat(0:dt:(n_steps-1)*dt, n_data, 1);
errors = zeros(n_grids, n_steps);
figure();
for k = 1:n_grids
    train_steps = train_steps_list(k);
    filename = [data_dir '/', data_kind, '/', data_kind, 'result_train', num2str(train_steps), '_node', num2str(approx_reservoir_size) '-L' num2str(L) '-radius' num2str(rho) '-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter', num2str(which_index_iter) '.mat'];
    load(filename, 'error');
    errors(k, :) = error(1, 1:n_steps);
    hold on;
    plot(times(1,:), error(1, 1:n_steps), 'DisplayName', ['train steps=' num2str(train_steps)]); 
    hold off;
end
sgtitle(['L=' num2str(L) ', g=' num2str(request_pool_size) ', rho=' num2str(rho) ', D_r=' num2str(approx_reservoir_size)])
legend();
