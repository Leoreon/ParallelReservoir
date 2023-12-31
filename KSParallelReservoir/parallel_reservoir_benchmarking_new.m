%Author: Jaideep Pathak, PhD candidate at the University of Maryland,
%College Park.
%email: jpathak@umd.edu or jaideeppathak244@gmail.com

clear

% function jobid = parallel_reservoir_benchmarking(request_pool_size)
 data_dir = './';
 % request_pool_size_list = 2:6;
 % request_pool_size_list = 1;
 % request_pool_size_list = 2;
 % request_pool_size_list = 4;
 % request_pool_size_list = 8;
 % request_pool_size_list = 16;
 % request_pool_size_list = 1:8;
 request_pool_size_list = 4:8;
 % request_pool_size = 8;
 % index_file = matfile('/lustre/jpathak/KS100/testing_ic_indexes.mat');
 index_file = matfile([data_dir 'testing_ic_indexes.mat']);
 %index_file = matfile('KS100/testing_ic_indexes.mat');

 full_pred_marker_array = index_file.testing_ic_indexes;
 
 num_indices = length(full_pred_marker_array);
 
 num_divided_jobs = 1;
 % num_divided_jobs = 2;
 % num_divided_jobs = 10;
 
 indices_per_job = num_indices/num_divided_jobs;
 
 % locality_list = 6:2:8;
 % locality_list = 7:8;
 % locality_list = 0;
 locality_list = 18;
 % locality_list = 12;
 % locality_list = 16;
 % locality_list = [1 2 3 4 5 6 7 8 12];
 % locality_list = [3 4 5 6 7 8];
 % locality_list = [16 15 14 13 12];
 % locality_list = [26 25 24];
 % locality_list = [28 27];
 % locality_list = [30 31 32];
 % locality_list = [33 34 35];
 % locality_list = 2;
 % locality_list = 8;
 % locality_list = 0;
 % locality_list = 3;
 % locality_list = 8:-2:2;
 
 % rho_list = 0.05:0.05:1.5;
 % rho_list = 0.9:0.1:1.7;
 % rho_list = 1.7:0.1:2.0;
 % rho_list = 1.41:0.04:1.79;
 % rho_list = 0.75;
 % rho_list = 0.6:0.2:1.6;
 % rho_list = [1.1 1.2 1.3 1.4 1.5 1.6];
 % rho_list = 1.4;
 % rho_list = 1.5:0.1:1.7;
 % rho_list = 0.8;
 rho_list = 0.6;
 % rho_list = 1.6;

 % train_steps_list = [2e4+1];
 % train_steps_list = 1.1e4;
 train_steps_list = 8e4;
 % train_steps_list = [2e4+1 4e4 6e4];
 % rho_list = 0.2:1:1.7;
 % locality_list = 3:4:8;
 h = waitbar(0,'Please wait...');
 progress = 0;
 for request_pool_size = request_pool_size_list
 for index_iter = 1:1
 % for index_iter = 2:10
 for rho = rho_list
 for locality_ = locality_list
 for train_steps = train_steps_list
    tic;
    partial_pred_marker_array = full_pred_marker_array((index_iter-1)*indices_per_job + 1:index_iter*indices_per_job);

    pred_marker_array = Composite(request_pool_size);
    which_index_iter = Composite(request_pool_size);
    rho_array = Composite(request_pool_size);
    locality_array = Composite(request_pool_size);
    
 
    for i=1:length(pred_marker_array)
        pred_marker_array{i} = partial_pred_marker_array;
        which_index_iter{i} = index_iter;
        rho_array{i} = rho;
        locality_array{i} = locality_;
    end

    request_pool_size
    spmd(request_pool_size)
        jobid = 2;
        % jobid = 1;
        
        data_kind = 'KS';
        % data_kind = 'NLKS';
        % data_kind = 'CGL';
        % data_kind = 'NLCGL';
        switch data_kind
            case 'CGL'
                % L = 44; 
                % L = 36; 
                % L = 30; 
                % L = 26; 
                L = 22; 
                % L = 18; 
                % L = 14; 
                % L = 8; 
                N = 32;
                % N = 64;
                % N = 128;
                test_steps = 20000;
                % train_steps = 80000;
                % train_steps = 100000;
                % c1 = -1.05; c2 = 1.05;
                c1 = -2; c2 = 2;
                % train_steps = 100000;dps  = 700000;  %50 200     % Number of stored times
 
                % train_steps = 200000; 
                % L = 8; N = 64; train_steps = 80000; test_steps = 20000;
                max_lyapunov = 1.0;
                % m = matfile([data_dir 'CGL_L', num2str(L) '_N_', num2str(N) '_dps', num2str(train_steps) '.mat']); % CGL
                % tf = matfile([data_dir 'CGL_L', num2str(L) '_N_' num2str(N) '_dps' num2str(test_steps) '.mat']); % CGL
                
                % iter = true;
                iter = false;

                if iter % iter8
                    n_iter = 7;
                    % n_iter = 8;
                    m = matfile([data_dir 'CGL_iter' num2str(n_iter) '_L', num2str(L) '_N_', num2str(N) '_dps', num2str(train_steps) 'c1_' num2str(c1) 'c2_' num2str(c2) '.mat']); % CGL
                else % iter1
                    m = matfile([data_dir 'CGL_L', num2str(L) '_N_', num2str(N) '_dps', num2str(train_steps) 'c1_' num2str(c1) 'c2_' num2str(c2) '.mat']); % CGL
                end
                % tf = matfile([data_dir 'CGL_test_L', num2str(L) '_N_' num2str(N) '_dps' num2str(test_steps) 'c1_' num2str(c1) 'c2_' num2str(c2) '.mat']); % CGL
            
                tf = matfile([data_dir 'CGL_L', num2str(L) '_N_' num2str(N) '_dps' num2str(test_steps) 'c1_' num2str(c1) 'c2_' num2str(c2) '.mat']); % CGL
            case 'KS'
                % L = 22; N = 64; 
                L = 22; N = 840;
                % L = 25; N = 64;
                % L = 50; N = 128; 
                % L = 50; N = 512; 
                % L = 50; N = 1024; 
                % L = 50; N = 2048; 
                % L = 100; N = 256; 
                train_steps = 80000; test_steps = 20000; 
                iter = false;
                switch L
                    case 22
                        max_lyapunov = 0.0825;
                    case 25
                        max_lyapunov = 0.0877;
                    case 50
                        max_lyapunov = 0.0743;
                    case 100
                        max_lyapunov = 0.09;
                    case 200
                        max_lyapunov = 0.09;
                end

                m = matfile([data_dir data_kind '_L', num2str(L) '_N_', num2str(N) '_dps', num2str(train_steps) '.mat']); % CGL
                tf = matfile([data_dir data_kind '_L', num2str(L) '_N_', num2str(N) '_dps', num2str(test_steps) '.mat']); % CGL
            
                % m = matfile([data_dir 'train_input_sequence.mat']); % KS
                % tf = matfile([data_dir 'test_input_sequence.mat']); % KS
            case 'NLKS'
                % L = 22; N = 64; train_steps = 80000; test_steps = 20000; 
                L = 50; N = 128; train_steps = 80000; test_steps = 20000; 
                iter = false;
                max_lyapunov = 0.0825;

                m = matfile([data_dir data_kind '_L', num2str(L) '_N_', num2str(N) '_dps', num2str(train_steps) '.mat']); % CGL
                tf = matfile([data_dir data_kind '_L', num2str(L) '_N_', num2str(N) '_dps', num2str(test_steps) '.mat']); % CGL
            
                % m = matfile([data_dir 'train_input_sequence_nonlocal.mat']); % KS
                % tf = matfile([data_dir 'test_input_sequence_nonlocal.mat']); % KS
            case 'NLCGL'
                % L = 44; 
                % L = 36; 
                % L = 30; 
                % L = 26; 
                % L = 22; 
                L = 18; 
                % L = 14; 
                % L = 8; 
                N = 32;
                % N = 64;
                % N = 128;
                test_steps = 20000;
                % train_steps = 80000;
                % train_steps = 100000;
                % c1 = -1.05; c2 = 1.05;
                c1 = -2; c2 = 2;
                % train_steps = 100000;dps  = 700000;  %50 200     % Number of stored times
 
                % train_steps = 200000; 
                % L = 8; N = 64; train_steps = 80000; test_steps = 20000;
                max_lyapunov = 1.0;
                % m = matfile([data_dir 'CGL_L', num2str(L) '_N_', num2str(N) '_dps', num2str(train_steps) '.mat']); % CGL
                % tf = matfile([data_dir 'CGL_L', num2str(L) '_N_' num2str(N) '_dps' num2str(test_steps) '.mat']); % CGL
                
                % iter = true;
                iter = false;

                if iter % iter8
                    n_iter = 7;
                    % n_iter = 8;
                    m = matfile([data_dir data_kind '_iter' num2str(n_iter) '_L', num2str(L) '_N_', num2str(N) '_dps', num2str(train_steps) 'c1_' num2str(c1) 'c2_' num2str(c2) '.mat']); % CGL
                else % iter1
                    m = matfile([data_dir data_kind '_L', num2str(L) '_N_', num2str(N) '_dps', num2str(train_steps) 'c1_' num2str(c1) 'c2_' num2str(c2) '.mat']); % CGL
                end
                tf = matfile([data_dir data_kind '_L', num2str(L) '_N_' num2str(N) '_dps' num2str(test_steps) 'c1_' num2str(c1) 'c2_' num2str(c2) '.mat']); % CGL
            
        end
        % fprintf(['use ', data_kind, '\n']);
        % m = matfile('/lustre/jpathak/KS100/train_input_sequence.mat'); 
%        m = matfile('KS100/train_input_sequence.mat');
%        tf = matfile('KS100/test_input_sequence.mat');
        % tf = matfile('/lustre/jpathak/KS100/test_input_sequence.mat');
        
        % sigma = 0.5;  %% simple scaling of data by a scalar
        sigma = 0.5;  %% simple scaling of data by a scalar
        
        train_uu = m.train_input_sequence;
        % [len, num_inputs] = size(m, 'train_input_sequence');
        % clear m;
        m = [];
        [len, num_inputs] = size(train_uu);

        num_workers = numlabs; %numlabs is a matlab func that returns the number of workers allocated. equal to request_pool_size

        chunk_size = num_inputs/numlabs; %%%%%%%%%% MUST DIVIDE (each reservoir responsible for this chunk)
        % fprintf('chunk size: %f\n', chunk_size);
        % fprintf('num_inputs: %f, numlabs: %f\n', num_inputs, numlabs);
        l = labindex; % labindex is a matlab function that returns the worker index

        chunk_begin = chunk_size*(l-1)+1;

        chunk_end = chunk_size*l;

        % locality = 6; % there are restrictions on the allowed range of this parameter. check documentation
        locality = locality_array; % there are restrictions on the allowed range of this parameter. check documentation
   
        rear_overlap = indexing_function_rear(chunk_begin, locality, num_inputs, chunk_size);  %spatial overlap on the one side
        
        forward_overlap = indexing_function_forward(chunk_end, locality, num_inputs, chunk_size);  %spatial overlap on the other side
        
        overlap_size = length(rear_overlap) + length(forward_overlap); 

        approx_reservoir_size = 20160 / num_workers;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 5000;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 7000 * 8 / request_pool_size;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 7000;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 9000;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 12000;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)

        avg_degree = 3; %average connection degree

        resparams.sparsity = avg_degree/approx_reservoir_size;
        
        resparams.degree = avg_degree;
        
        nodes_per_input = round(approx_reservoir_size/(chunk_size+overlap_size));

        resparams.N = nodes_per_input*(chunk_size+overlap_size); % exact number of nodes in the network

        resparams.discard_length = 1000;  %number of time steps used to discard transient (generously chosen)

        resparams.train_length = train_steps-resparams.discard_length;  %number of time steps used for training
        % resparams.train_length = 99000;  %number of time steps used for training
        % resparams.train_length = 79000;  %number of time steps used for training
        % resparams.train_length = 39000;  %number of time steps used for training

        % sync_length = 32; % use a short time series to synchronize to the data at the prediction_marker
        sync_length = 100; % use a short time series to synchronize to the data at the prediction_marker
        
        % resparams.predict_length = 2999;  %number of steps to be predicted
        % resparams.predict_length = test_steps-sync_length-1;  %number of steps to be predicted
        % resparams.predict_length = test_steps-sync_length-pred_marker_array(end);  %number of steps to be predicted
        resparams.predict_length = 3000-sync_length-1;  %number of steps to be predicted

        % resparams.radius = 0.6; % spectral radius of the reservoir
        resparams.radius = rho; % spectral radius of the reservoir
        
        resparams.beta = 0.0001; % ridge regression regularization parameter
        
        u = zeros(len, chunk_size + overlap_size); % this will be populated by the input data to the reservoir

        u(:,1:locality) = train_uu(1:end, rear_overlap);

        u(:,locality+1:locality+chunk_size) = train_uu(1:end, chunk_begin:chunk_end);

        u(:,locality+chunk_size+1:2*locality+chunk_size) = train_uu(1:end,forward_overlap);

        u = sigma*u;
        % [x, w_out, w, w_in, RMSE] = train_reservoir(resparams, in, labindex, jobid, locality, chunk_size);
        
        train_uu = [];

        test_uu = tf.test_input_sequence;
        % clear tf;

        test_u = zeros(20000, chunk_size + overlap_size); % this will be populated by the input data to the reservoir

        test_u(:,1:locality) = test_uu(1:end, rear_overlap);

        test_u(:,locality+1:locality+chunk_size) = test_uu(1:end, chunk_begin:chunk_end);

        test_u(:,locality+chunk_size+1:2*locality+chunk_size) = test_uu(1:end,forward_overlap);

        test_u = sigma*test_u;

        tf = [];
        test_uu = [];
        % fprintf('start res train predict');
        % [pred_collect, RMSE] = res_predict(x, w_out, w, w_in, transpose(test_u), resparams, jobid, locality, chunk_size, pred_marker_array, sync_length);
        
        fprintf('start learning\n');
        % learn = 'RLS';
        learn = 'LSM';
        learn = 'LSM_common';
        switch learn
            case 'LSM'
                if iter
                    [pred_collect, RMSE] = res_train_predict_iter(transpose(u), transpose(test_u), resparams, jobid, locality, chunk_size, pred_marker_array, sync_length);
                else
                    [pred_collect, RMSE] = res_train_predict(transpose(u), transpose(test_u), resparams, jobid, locality, chunk_size, pred_marker_array, sync_length);
                end
            case 'RLS'
                [num_inputs2,~] = size(u.');
                A = generate_reservoir(resparams.N, resparams.radius, resparams.degree, labindex, jobid);
                q = resparams.N/num_inputs2;
                win = zeros(resparams.N, num_inputs2);
                for i=1:num_inputs2
                    rng(i)
                    ip = (-1 + 2*rand(q,1));
                    win((i-1)*q+1:i*q,i) = ip;
                end
                
                wout = zeros(chunk_size, resparams.N);
                [x, wout] = recursive_least_square(resparams, u.', win, A, wout, locality, chunk_size, sync_length);
                pred_collect = res_predict(x, wout, A, win, transpose(test_u), resparams, jobid, locality, chunk_size, pred_marker_array, sync_length);
            case 'LMS_common'
                switch l
                    case 1
                        [num_inputs2,~] = size(u.');
                        A = generate_reservoir(resparams.N, resparams.radius, resparams.degree, labindex, jobid);
                        q = resparams.N/num_inputs2;
                        win = zeros(resparams.N, num_inputs2);
                        for i=1:num_inputs2
                            rng(i)
                            ip = (-1 + 2*rand(q,1));
                            win((i-1)*q+1:i*q,i) = ip;
                        end
                        wout = zeros(chunk_size, resparams.N);
                        [x, wout] = recursive_least_square(resparams, u.', win, A, wout, locality, chunk_size, sync_length);
                        destinations = 2:num_workers;
                        labSend(A, destinations);
                        labBarrier;
                        labSend(win, destinations);
                        labSend(wout, destinations);
                    otherwise
                        A = labReceive();
                        win = labReceive();
                        wout = labReceive();
                end
                pred_collect = res_predict(x, wout, A, win, transpose(test_u), resparams, jobid, locality, chunk_size, pred_marker_array, sync_length);
        end
        % fprintf('pred_collect');
        collated_prediction = gcat(pred_collect,1,1);
    end
    
    runtime = toc
    approx_reservoir_size = approx_reservoir_size{1};
    locality = locality{1};
    num_workers = num_workers{1};
    jobid = jobid{1};
    % data_file = m{1};
    % test_file = tf{1};
    data_kind = data_kind{1};
    max_lyapunov = max_lyapunov{1};
    train_steps = train_steps{1}; test_steps = test_steps{1};
    L = L{1};
    switch data_kind
        case 'CGL' 
            dt = 0.07;
            N = N{1}; c1 = c1{1}; c2 = c2{1};
            % data_file = load([data_dir 'CGL_L18_N_32_dps80000.mat']); % CGL
            test_file = load([data_dir data_kind '_L' num2str(L) '_N_' num2str(N) '_dps' num2str(test_steps) 'c1_' num2str(c1) 'c2_' num2str(c2) '.mat']); % CGL
            % test_file = load([data_dir 'CGL_L' num2str(L) '_N_' num2str(N) '_dps' num2str(test_steps) '.mat']); % CGL
        case 'KS'
            dt = 1/4;
            N = N{1}; 
            % data_file = load([data_dir 'train_input_sequence.mat']); % KS
            % test_file = load([data_dir 'test_input_sequence.mat']); % KS
            test_file = matfile([data_dir data_kind '_L', num2str(L) '_N_', num2str(N) '_dps', num2str(test_steps) '.mat']); % KS
        case 'NLKS'
            dt = 1/4;
            test_file = matfile([data_dir 'test_input_sequence_nonlocal.mat']); % KS
        case 'NLCGL'
            dt = 0.07;
            N = N{1}; c1 = c1{1}; c2 = c2{1};
            % data_file = load([data_dir 'CGL_L18_N_32_dps80000.mat']); % CGL
            test_file = load([data_dir data_kind '_L' num2str(L) '_N_' num2str(N) '_dps' num2str(test_steps) 'c1_' num2str(c1) 'c2_' num2str(c2) '.mat']); % CGL
            % test_file = load([data_dir 'CGL_L' num2str(L) '_N_' num2str(N) '_dps' num2str(test_steps) '.mat']); % CGL
    end
    resparams = resparams{1};
    sync_length = sync_length{1};
    sigma = sigma{1};
    chunk_size = chunk_size{1};
    num_inputs = num_inputs{1};
    pred_collect = collated_prediction{1};
    pred_marker_array = pred_marker_array{1};
    num_preds = length(pred_marker_array);
    diff = zeros(num_inputs, num_preds*resparams.predict_length);
    trajectories_true = zeros(num_inputs, num_preds*resparams.predict_length);
    
    switch learn{1} 
        case 'LSM'
            RMSE_mean = 0;
            for i = 1:num_workers
                RMSE_mean = RMSE_mean + RMSE{i};
                i;
            end
            RMSE_mean = RMSE_mean / num_workers;
        case 'RLS'
            RMSE_mean = 0;
    end

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
    % filename = [data_dir '/', data_kind, '/', data_kind, 'result_uniform_train', num2str(train_steps), '_node', num2str(approx_reservoir_size) '-L' num2str(L) '-radius' num2str(rho) '-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter', num2str(which_index_iter) '.mat'];
    filename = [data_dir '/', data_kind, '/', data_kind, 'result_linear_train', num2str(train_steps), '_node', num2str(approx_reservoir_size) '-L' num2str(L) '-radius' num2str(rho) '-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter', num2str(which_index_iter) '.mat'];
    % filename = [data_dir '/', data_kind, '/', data_kind '100-' num2str(approx_reservoir_size) '-L' num2str(L) '-radius' num2str(rho) '-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter', num2str(which_index_iter) '.mat'];
    save(filename, 'pred_collect', 'error', 'diff', 'resparams', 'RMSE_mean', 'pred_marker_array', 'trajectories_true', 'locality', 'chunk_size');
    display(filename);
    
    n_steps = size(trajectories_true, 2);
    n_data = size(trajectories_true, 1);
    times = repmat(0:dt*max_lyapunov:(n_steps-1)*dt*max_lyapunov, n_data, 1);
    locations = repmat((1:n_data).', 1, n_steps);
    max_value = max(max(trajectories_true)); min_value = min(min(trajectories_true));
    figure(); 
    subplot(3, 1, 1); surf(times, locations, [trajectories_true(1:2:end-1,1:n_steps); trajectories_true(2:2:end,1:n_steps)]); view(0, 90); shading interp, axis tight; xlabel('lyapunov time'); ylabel('x'); title('true data'); colorbar; clim([min_value max_value]); xlim([0 50]);
    subplot(3, 1, 2); surf(times, locations, [pred_collect(1:2:end-1,:); pred_collect(2:2:end,:)]); view(0, 90); shading interp, axis tight; xlabel('lyapunov time'); ylabel('x'); title('predicted data'); colorbar; clim([min_value max_value]); xlim([0 50]);
    subplot(3, 1, 3); surf(times, locations, [diff(1:2:end-1,:); diff(2:2:end, :)]); view(0, 90); shading interp, axis tight; xlabel('lyapunov time'); ylabel('x'); title('error'); colorbar; clim(2 * [min_value max_value]); xlim([0 50]);
    sgtitle([data_kind ' L:' num2str(L) ' rho: ' num2str(resparams.radius) ', request pool size: ' num2str(request_pool_size) ', locality: ' num2str(locality)]);
    
    figure(); plot(times(1,:), sqrt(mean(diff.^2, 1)));
    xlabel('time (lyapunov*second'); ylabel('RMSE'); title('error');

    progress = progress + 1;
    total = size(rho_list, 2) * size(locality_list, 2);
    h = waitbar(progress/total,h,... 
    sprintf('progress: %d/%d', progress, total));
    close all;

    clear pred_marker_array which_index_iter rho_array locality_array;
 end
 end
 end
 end
 end
 close(h);
% end
 




