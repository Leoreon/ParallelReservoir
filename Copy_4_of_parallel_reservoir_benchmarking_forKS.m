%Author: Jaideep Pathak, PhD candidate at the University of Maryland,
%College Park.
%email: jpathak@umd.edu or jaideeppathak244@gmail.com
%% 二章　空間特性からのハイパーパラメータの決定
% Parallel and Spatial
clear

% function jobid = parallel_reservoir_benchmarking(request_pool_size)


% request_pool_size_list = 1;
% request_pool_size_list = 2;
% request_pool_size_list = 3;
% request_pool_size_list = 4;
% request_pool_size_list = 5;
% request_pool_size_list = 6;
% request_pool_size_list = 7;
% request_pool_size_list = 8;
% request_pool_size_list = 10;
% request_pool_size_list = 11;
% request_pool_size_list = 12;
% request_pool_size_list = 14;
% request_pool_size_list = 15;
% request_pool_size_list = 16;
request_pool_size_list = 2:8;
% request_pool_size_list = 1:8;
% request_pool_size_list = 2:8;
% request_pool_size_list = [3 5 7];
% request_pool_size_list = 4:2:16;
% request_pool_size_list = 14:2:16;
% request_pool_size_list = 16:-2:2;
% request_pool_size_list = [2:8 10 12 14 15];
% request_pool_size_list = [12 14 15];
% request_pool_size_list = 2:6;
% request_pool_size_list = 8:-1:5;
% request_pool_size_list = 4;
% request_pool_size = 8;
for request_pool_size = request_pool_size_list
num_reservoirs = request_pool_size;
num_reservoirs_per_worker = 1;
% show_fig = true;
show_fig = false;
data_dir = './';
% index_file = matfile('/lustre/jpathak/KS100/testing_ic_indexes.mat');
index_file = matfile([data_dir 'testing_ic_indexes.mat']);
%index_file = matfile('KS100/testing_ic_indexes.mat');

full_pred_marker_array = index_file.testing_ic_indexes;

num_indices = length(full_pred_marker_array);

num_divided_jobs = 1;
% num_divided_jobs = 2;
% num_divided_jobs = 10;

indices_per_job = num_indices/num_divided_jobs;

% locality_list = 0;
% locality_list = 1;
% locality_list = 2;
% locality_list = 3;
% locality_list = 5;
% locality_list = 6;
% locality_list = 8;
% locality_list = 9;
% locality_list = 12;
% locality_list = 16;
% locality_list = 18;
% locality_list = 20;
% locality_list = 30;
% locality_list = 35;
% locality_list = 40;
% locality_list = 50;
% locality_list = 60;
% locality_list = 65;
% locality_list = 70;
% locality_list = 80;
% locality_list = 90;
% locality_list = 100;
% locality_list = 120;
% locality_list = 160;
% locality_list = 200;
% locality_list = 1:2:15;
% locality_list = 2:2:15;
% locality_list = [100 150 200];
% locality_list = 15;
% locality_list = [70];
% locality_list = [100];
% locality_list = [10 15 20 25 30 35 40 45];
% locality_list = [25 30 35 40 45];
% locality_list = 10:10:80;
% locality_list = 90:10:160;
locality_list = 20:20:280;
% locality_list = 20:20:160;
% locality_list = 160:-20:20;
% locality_list = 180:20:280;
% locality_list = 240:20:280;
% locality_list = 40:20:100;
% locality_list = 80:20:160;
% locality_list = [50 55];
% locality_list = 20:5:60;
% locality_list = 7:8;
% locality_list = [9 10 11 12];
% locality_list = 6:2:8;
% locality_list = 16:-2:8;
% locality_list = 8:-2:2;
% locality_list = 0:200;
% locality_list = [130:140];
% locality_list = [1 2 3 4 5 6 7 8 12];
% locality_list = [3 4 5 6 7 8];
% locality_list = [16 15 14 13 12];
% locality_list = [26 25 24];
% locality_list = [28 27];
% locality_list = [30 31 32];
% locality_list = [33 34 35];

% width_list = 0;
% width_list = 1;
% width_list = 3;
% width_list = 5;
% width_list = [11:19];
% width_list = 0:4:20;
% width_list = 0:2:60;
% width_list = 62:2:80;
% width_list = 24:4:120;
% width_list = 0:5:20;
% width_list = 25:5:60;
% width_list = 1:4:20;
% width_list = 10:20:30;
% width_list = 10:10:50;
% width_list = 0:100:2500;
% width_list = 20;
% width_list = 400;
width_list = 1e7;

rho_list = 0.6;
% rho_list = 0.75;
% rho_list = 0.8;
% rho_list = 0.9;
% rho_list = 1.5;
% rho_list = 1.6;
% rho_list = 0.05:0.05:1.5;
% rho_list = 0.9:0.1:1.7;
% rho_list = 1.7:0.1:2.0;
% rho_list = 1.41:0.04:1.79;
% rho_list = 0.6:0.2:1.6;
% rho_list = 0.1:0.2:1.7;
% rho_list = [1.1 1.2 1.3 1.4 1.5 1.6];
% rho_list = 1.5:0.1:1.7;

% train_steps_list = [2e4+1];
% train_steps_list = 1.1e4;
train_steps_list = 8e4;
% train_steps_list = 15e4;
% train_steps_list = 30e4;
% train_steps_list = [2e4+1 4e4 6e4];
% rho_list = 0.2:1:1.7;
% locality_list = 3:4:8;
jobid_list = 1;
% jobid_list = 1:2;
% jobid_list = 1:3;
% jobid_list = 4;
% jobid_list = 2:3;
% jobid_list = 2:5;
% jobid_list = 1:5;
% jobid_list = 6:10;
% jobid_list = 5;
% jobid_list = [1 2 5];
h = waitbar(0,'Please wait...');
progress = 0;
% for request_pool_size = request_pool_size_list
for index_iter = 1:1
% for index_iter = 2:10
for rho = rho_list
for locality_ = locality_list
for width = width_list
for train_steps = train_steps_list
for jobid = jobid_list
    tic;
    % partial_pred_marker_array = full_pred_marker_array((index_iter-1)*indices_per_job + 1:index_iter*indices_per_job);
    % partial_pred_marker_array = full_pred_marker_array(1);
    % partial_pred_marker_array = full_pred_marker_array(1:20);
    partial_pred_marker_array = full_pred_marker_array(1:20);
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
    
    % [num_reservoirs_per_worker, num_rem] = quorem(num_reservoirs, request_pool_size);
    num_rem = mod(num_reservoirs, request_pool_size);
    num_reservoirs_per_worker = floor(num_reservoirs/request_pool_size);
    request_pool_size
    spmd(request_pool_size)
        % jobid = 3;
        % jobid = 1;
        
        % data_kind = 'KS';
        % data_kind = 'KS_slow';
        data_kind = 'KS_slow_short';
        % data_kind = 'NLKS';
        % data_kind = 'CGL';
        % data_kind = 'NLCGL';
        % data_kind = 'LCD';
        % data_kind = 'KS_sync';
        switch data_kind
            case 'LCD'
                pred_marker_array = pred_marker_array(1);
                iter = false;
                n_kind_data = 6;
                n_kind_input = 1;
                max_lyapunov = 1;
                % L = 11;
                train_steps = 16650;
                test_steps = 3330;
                % load_ekisho();
                train_dir = 'LCD_data/data/train/';
                test_dir = 'LCD_data/data/test/';
                % % nx = load(strcat(test_dir, 'nx.dat'));
                %% load data of Liquid Crystal Display
                train_filename = [train_dir 'LCD_data.mat'];
                test_filename = [test_dir 'LCD_data.mat'];
                m = matfile([train_filename]); % KS
                tf = matfile([test_filename]); % KS
            
                L = m.L; sample_dT = m.sample_dT; N = m.N;
                train_input = m.Input; 
                test_input = tf.Input;
            case 'KS'
                n_kind_data = 1; n_kind_input = 0;
                % L = 22; N = 64; 
                L = 22; N = 840;
                % L = 26; N = 840;
                % L = 44; N = 832;
                % L = 44; N = 840;
                % L = 44; N = 1680;
                % L = 50; N = 840;
                % L = 66; N = 840;
                % L = 88; N = 924;
                % L = 88; N = 840;
                % L = 100; N = 840;
                % L = 25; N = 64;
                % L = 50; N = 120;
                % L = 52; N = 128; 
                % L = 52; N = 192; 
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
                    case 26
                        max_lyapunov = 0.0526; % tekitou
                        % max_lyapunov = 0.0877; % tekitou
                    case 44
                        max_lyapunov = 0.09;
                    case 50
                        max_lyapunov = 0.0743;
                    case 52
                        max_lyapunov = 0.0776;
                    case 66
                        max_lyapunov = 0.0856;
                    case 88
                        max_lyapunov = 0.0886;
                    case 100
                        max_lyapunov = 0.09;
                    case 200
                        max_lyapunov = 0.09;
                end

                m = matfile([data_dir data_kind '_L', num2str(L) '_N_', num2str(N) '_dps', num2str(train_steps) '.mat']); % KS
                tf = matfile([data_dir data_kind '_L', num2str(L) '_N_', num2str(N) '_dps', num2str(test_steps) '.mat']); % KS
            
                % m = matfile([data_dir 'train_input_sequence.mat']); % KS
                % tf = matfile([data_dir 'test_input_sequence.mat']); % KS
                train_input = []; test_input = [];
            case 'KS_slow'
                n_kind_data = 1; n_kind_input = 0;
                % L = 22; N = 64; 
                L = 22; N = 840;
                % L = 26; N = 840;
                % L = 44; N = 832;
                % L = 44; N = 840;
                % L = 44; N = 1680;
                % L = 50; N = 840;
                % L = 66; N = 840;
                % L = 88; N = 924;
                % L = 88; N = 840;
                % L = 100; N = 840;
                % L = 25; N = 64;
                % L = 50; N = 120;
                % L = 52; N = 128; 
                % L = 52; N = 192; 
                % L = 50; N = 512; 
                % L = 50; N = 1024; 
                % L = 50; N = 2048; 
                % L = 100; N = 256; 
                % train_steps = 160000; test_steps = 40000; 
                train_steps = 80000; test_steps = 20000; 
                iter = false;
                switch L
                    case 22
                        max_lyapunov = 0.0825;
                    case 25
                        max_lyapunov = 0.0877;
                    case 26
                        max_lyapunov = 0.0526; % tekitou
                        % max_lyapunov = 0.0877; % tekitou
                    case 44
                        max_lyapunov = 0.09;
                    case 50
                        max_lyapunov = 0.0743;
                    case 52
                        max_lyapunov = 0.0776;
                    case 66
                        max_lyapunov = 0.0856;
                    case 88
                        max_lyapunov = 0.0886;
                    case 100
                        max_lyapunov = 0.09;
                    case 200
                        max_lyapunov = 0.09;
                end

                m = matfile([data_dir data_kind '_L', num2str(L) '_N_', num2str(N) '_dps', num2str(train_steps) '.mat']); % KS
                tf = matfile([data_dir data_kind '_L', num2str(L) '_N_', num2str(N) '_dps', num2str(test_steps) '.mat']); % KS
            
                % m = matfile([data_dir 'train_input_sequence.mat']); % KS
                % tf = matfile([data_dir 'test_input_sequence.mat']); % KS
                train_input = []; test_input = [];
            case 'KS_slow_short'
                n_kind_data = 1; n_kind_input = 0;
                % L = 22; N = 64; 
                L = 22; N = 840;
                % L = 26; N = 840;
                % L = 44; N = 832;
                % L = 44; N = 840;
                % L = 44; N = 1680;
                % L = 50; N = 840;
                % L = 66; N = 840;
                % L = 88; N = 924;
                % L = 88; N = 840;
                % L = 100; N = 840;
                % L = 25; N = 64;
                % L = 50; N = 120;
                % L = 52; N = 128; 
                % L = 52; N = 192; 
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
                    case 26
                        max_lyapunov = 0.0526; % tekitou
                        % max_lyapunov = 0.0877; % tekitou
                    case 44
                        max_lyapunov = 0.09;
                    case 50
                        max_lyapunov = 0.0743;
                    case 52
                        max_lyapunov = 0.0776;
                    case 66
                        max_lyapunov = 0.0856;
                    case 88
                        max_lyapunov = 0.0886;
                    case 100
                        max_lyapunov = 0.09;
                    case 200
                        max_lyapunov = 0.09;
                end

                m = matfile([data_dir data_kind '_L', num2str(L) '_N_', num2str(N) '_dps', num2str(train_steps) '.mat']); % KS
                tf = matfile([data_dir data_kind '_L', num2str(L) '_N_', num2str(N) '_dps', num2str(test_steps) '.mat']); % KS
            
                % m = matfile([data_dir 'train_input_sequence.mat']); % KS
                % tf = matfile([data_dir 'test_input_sequence.mat']); % KS
                train_input = []; test_input = [];
            case 'KS_sync'
                n_kind_data = 1; n_kind_input = 1;
                L = 22; N = 64; 
                % L = 22; N = 840;
                % L = 26; N = 840;
                % L = 44; N = 832;
                % L = 44; N = 840;
                % L = 50; N = 840;
                % L = 66; N = 840;
                % L = 88; N = 924;
                % L = 88; N = 840;
                % L = 100; N = 840;
                % L = 25; N = 64;
                % L = 50; N = 120;
                % L = 52; N = 128; 
                % L = 52; N = 192; 
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
                    case 26
                        max_lyapunov = 0.0526; % tekitou
                        % max_lyapunov = 0.0877; % tekitou
                    case 44
                        max_lyapunov = 0.09;
                    case 50
                        max_lyapunov = 0.0743;
                    case 52
                        max_lyapunov = 0.0776;
                    case 66
                        max_lyapunov = 0.0856;
                    case 88
                        max_lyapunov = 0.0886;
                    case 100
                        max_lyapunov = 0.09;
                    case 200
                        max_lyapunov = 0.09;
                end

                m = matfile([data_dir data_kind '_L', num2str(L) '_N_', num2str(N) '_dps', num2str(train_steps) '.mat']); % KS
                tf = matfile([data_dir data_kind '_L', num2str(L) '_N_', num2str(N) '_dps', num2str(test_steps) '.mat']); % KS
            
                % m = matfile([data_dir 'train_input_sequence.mat']); % KS
                % tf = matfile([data_dir 'test_input_sequence.mat']); % KS
                train_input = m.Input; test_input = tf.Input;
            case 'CGL'
                n_kind_data = 2; n_kind_input = 0;
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
                c1 = 2; c2 = -2;
                % c1 = -2; c2 = 2;
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
                
                test_filename = [data_dir 'CGL_test_L', num2str(L) '_N_' num2str(N) '_dps' num2str(test_steps) 'c1_' num2str(c1) 'c2_' num2str(c2) '.mat'];
                tf = matfile(test_filename); % CGL
                train_input = []; test_input = [];
            case 'NLKS'
                n_kind_data = 1; n_kind_input = 0;
                % L = 22; N = 64; train_steps = 80000; test_steps = 20000; 
                L = 50; N = 128; train_steps = 80000; test_steps = 20000; 
                iter = false;
                max_lyapunov = 0.0825;

                m = matfile([data_dir data_kind '_L', num2str(L) '_N_', num2str(N) '_dps', num2str(train_steps) '.mat']); % CGL
                tf = matfile([data_dir data_kind '_L', num2str(L) '_N_', num2str(N) '_dps', num2str(test_steps) '.mat']); % CGL
            
                % m = matfile([data_dir 'train_input_sequence_nonlocal.mat']); % KS
                % tf = matfile([data_dir 'test_input_sequence_nonlocal.mat']); % KS
            case 'NLCGL'
                 n_kind_input = 0;
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
        
        sigma = 1.0;  %% simple scaling of data by a scalar
        % sigma = 0.5;  %% simple scaling of data by a scalar
        
        train_uu = m.train_input_sequence;
        % [len, num_inputs] = size(m, 'train_input_sequence');
        % clear m;
        m = [];
        [len, num_inputs_data] = size(train_uu);
        [len, num_inputs_input] = size(train_input);

        num_workers = numlabs; %numlabs is a matlab func that returns the number of workers allocated. equal to request_pool_size
        l = labindex; % labindex is a matlab function that returns the worker index
        
        % workerごとにリザバーの数が異なる時（今は使わない）
        if l <= num_rem 
            num_reservoirs_per_worker = num_reservoirs_per_worker + 1;
        end

        % chunk_size_data = num_inputs_data/num_workers; %%%%%%%%%% MUST DIVIDE (each reservoir responsible for this chunk)
        % chunk_size_input = num_inputs_input/num_workers; %%%%%%%%%% MUST DIVIDE (each reservoir responsible for this chunk)
        chunk_size_data = num_inputs_data/num_reservoirs; %%%%%%%%%% MUST DIVIDE (each reservoir responsible for this chunk)
        chunk_size_input = num_inputs_input/num_reservoirs; %%%%%%%%%% MUST DIVIDE (each reservoir responsible for this chunk)
        
        % fprintf('chunk size: %f\n', chunk_size);
        % fprintf('num_inputs: %f, numlabs: %f\n', num_inputs, numlabs);
        
        chunk_begin_data = num_reservoirs_per_worker*chunk_size_data*(l-1)+1;
        chunk_begin_input = num_reservoirs_per_worker*chunk_size_input*(l-1)+1;

        chunk_end_data = num_reservoirs_per_worker*chunk_size_data*l;
        chunk_end_input = num_reservoirs_per_worker*chunk_size_input*l;
        
        % reservoir_kind = 'spatial';
        reservoir_kind = 'uniform';
        
        % learn = 'RLS';
        % learn = 'LSM';
        learn = 'LSM_common';
        % learn = 'LSM_GD_training_error';
        % learn = 'LSM_GD_short_prediction_time';
        % Gradient Descentの時はlocalityを最大に、時間はworkerごとに変える
        switch learn
            case 'LSM_GD_short_prediction_time'
                locality = floor((num_inputs_data-num_reservoirs_per_worker*chunk_size_data) / 2);
                % u_length = floor(train_steps/2);
                u_length = train_steps;
                % train_start = 1;
                train_start = 1 + (l-1)*(train_steps-u_length)/numlabs;
            case 'LSM_GD'
                % dl = rem(l, 2);
                locality = floor((num_inputs_data-num_reservoirs_per_worker*chunk_size_data) / 2);
                % resparams.train_length = floor(resparams.train_length / num_workers);
                u_length = floor(train_steps * (num_workers-1) / num_workers);
                train_start = floor((train_steps-u_length) / num_workers + 1);
            otherwise
                % locality = 6; % there are restrictions on the allowed range of this parameter. check documentation
                locality = locality_array; % there are restrictions on the allowed range of this parameter. check documentation
        
                % dl = 0;
                u_length = train_steps;
                train_start = 1;
        end
        % locality = locality + dl;

        resparams.discard_length = 1000;  %number of time steps used to discard transient (generously chosen)
        
        resparams.train_length = u_length-resparams.discard_length;  %number of time steps used for training
        % resparams.train_length = 99000;  %number of time steps used for training
        % resparams.train_length = 79000;  %number of time steps used for training
        % resparams.train_length = 39000;  %number of time steps used for training
        
        % sync_length = 32; % use a short time series to synchronize to the data at the prediction_marker
        % sync_length = 100; % use a short time series to synchronize to the data at the prediction_marker
        sync_length = 500; % use a short time series to synchronize to the data at the prediction_marker
        
        % resparams.predict_length = 2999;  %number of steps to be predicted
        % resparams.predict_length = test_steps-sync_length-1;  %number of steps to be predicted
        % resparams.predict_length = test_steps-sync_length-pred_marker_array(end);  %number of steps to be predicted
        resparams.predict_length = 3000-sync_length-1;  %number of steps to be predicted
        
        
        rear_overlap_data = indexing_function_rear(chunk_begin_data, n_kind_data*locality, num_inputs_data, chunk_size_data, data_kind, l);  %spatial overlap on the one side
        rear_overlap_input = indexing_function_rear(chunk_begin_input, n_kind_input*locality, num_inputs_input, chunk_size_input, data_kind, l);  %spatial overlap on the one side
        
        forward_overlap_data = indexing_function_forward(chunk_end_data, n_kind_data*locality, num_inputs_data, chunk_size_data, data_kind, l);  %spatial overlap on the other side
        forward_overlap_input = indexing_function_forward(chunk_end_input, n_kind_input*locality, num_inputs_input, chunk_size_input, data_kind, l);  %spatial overlap on the other side

        
        rear_locality_data = length(rear_overlap_data);
        rear_locality_input = length(rear_overlap_input);
        forward_locality_data = length(forward_overlap_data);
        forward_locality_input = length(forward_overlap_input);
        
        overlap_size_data = length(rear_overlap_data) + length(forward_overlap_data); 
        overlap_size_input = length(rear_overlap_input) + length(forward_overlap_input); 
        
        % approx_reservoir_size = 5040;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 192 / num_workers;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 256 / num_workers;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 1680 / num_workers;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 2520 / num_workers;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 3360 / num_workers;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        approx_reservoir_size = 5040 / num_workers;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 5880 / num_workers;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 5880 / num_reservoirs;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = (6720 + 840) / num_workers;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 7560 / num_workers;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 10080 / num_workers;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 15120 / num_workers;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 19968 / num_workers;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 20160 / num_workers;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 25200 / num_workers;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 30240 / num_workers;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 35280 / num_workers;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 40320 / num_workers;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 80016 / num_workers;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 5000;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 7000 * 8 / request_pool_size;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 7000;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 9000;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 9984;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
        % approx_reservoir_size = 12000;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)

        avg_degree = 3; %average connection degree

        resparams.sparsity = avg_degree/approx_reservoir_size;
        
        resparams.degree = avg_degree;
        
        % nodes_per_input = round(approx_reservoir_size/(chunk_size+overlap_size));
        % nodes_per_input = 1 + round(approx_reservoir_size/(chunk_size+overlap_size));
        nodes_per_input = round(approx_reservoir_size/(chunk_size_data+chunk_size_input));
        
        % 一つのリザバーのノード数
        % resparams.N = nodes_per_input*(chunk_size+overlap_size); % exact number of nodes in the network
        resparams.N = nodes_per_input*(chunk_size_data+chunk_size_input); % exact number of nodes in the network

        % resparams.radius = 0.6; % spectral radius of the reservoir
        resparams.radius = rho; % spectral radius of the reservoir
        
        resparams.beta = 0.0001; % ridge regression regularization parameter
        
        train_in = zeros(u_length, chunk_size_input + overlap_size_input); % this will be populated by the input data to the reservoir
        
        % train_in(:,1:n_kind_input*locality) = train_input(train_start:(n_kind_input~=0)*(train_start+u_length-1), rear_overlap_input);
        train_in(:,1:rear_locality_input) = train_input(train_start:(n_kind_input~=0)*(train_start+u_length-1), rear_overlap_input);
        
        % train_in(:,n_kind_input*locality+1:n_kind_input*locality+num_reservoirs_per_worker*chunk_size_input) = train_input(train_start:(n_kind_input~=0)*(train_start+u_length-1), chunk_begin_input:chunk_end_input);
        train_in(:,rear_locality_input+1:rear_locality_input+num_reservoirs_per_worker*chunk_size_input) = train_input(train_start:(n_kind_input~=0)*(train_start+u_length-1), chunk_begin_input:chunk_end_input);
        
        % train_in(:,n_kind_input*locality+num_reservoirs_per_worker*chunk_size_input+1:2*n_kind_input*locality+num_reservoirs_per_worker*chunk_size_input) = train_input(train_start:(n_kind_input~=0)*(train_start+u_length-1), forward_overlap_input);
        train_in(:,rear_locality_input+num_reservoirs_per_worker*chunk_size_input+1:rear_locality_input+num_reservoirs_per_worker*chunk_size_input+forward_locality_input) = train_input(train_start:(n_kind_input~=0)*(train_start+u_length-1), forward_overlap_input);
        
        train_in = sigma*train_in;

        u = zeros(u_length, num_reservoirs_per_worker*chunk_size_data + overlap_size_data); % this will be populated by the input data to the reservoir
        
        % u(:,1:n_kind_data*locality) = train_uu(train_start:train_start+u_length-1, rear_overlap_data);
        u(:,1:rear_locality_data) = train_uu(train_start:train_start+u_length-1, rear_overlap_data);
        
        % u(:,n_kind_data*locality+1:n_kind_data*locality+num_reservoirs_per_worker*chunk_size_data) = train_uu(train_start:train_start+u_length-1, chunk_begin_data:chunk_end_data);
        u(:,rear_locality_data+1:rear_locality_data+num_reservoirs_per_worker*chunk_size_data) = train_uu(train_start:train_start+u_length-1, chunk_begin_data:chunk_end_data);
        
        % u(:,n_kind_data*locality+num_reservoirs_per_worker*chunk_size_data+1:2*n_kind_data*locality+num_reservoirs_per_worker*chunk_size_data) = train_uu(train_start:train_start+u_length-1, forward_overlap_data);
        u(:,rear_locality_data+num_reservoirs_per_worker*chunk_size_data+1:rear_locality_data+num_reservoirs_per_worker*chunk_size_data+forward_locality_data) = train_uu(train_start:train_start+u_length-1, forward_overlap_data);
        
        u = sigma*u;
        % u = zeros(len, chunk_size + overlap_size); % this will be populated by the input data to the reservoir
        % 
        % u(:,1:locality) = train_uu(1:end, rear_overlap);
        % 
        % u(:,locality+1:locality+chunk_size) = train_uu(1:end, chunk_begin:chunk_end);
        % 
        % u(:,locality+chunk_size+1:2*locality+chunk_size) = train_uu(1:end,forward_overlap);
        % 
        % u = sigma*u;
        % [x, w_out, w, w_in, RMSE] = train_reservoir(resparams, in, labindex, jobid, locality, chunk_size);
        
        train_uu = [];

        test_in = zeros(size(test_input, 1), num_reservoirs_per_worker*chunk_size_input + overlap_size_input); % this will be populated by the input data to the reservoir
        
        % test_in(:,1:n_kind_input*locality) = test_input(1:end, rear_overlap_input);
        test_in(:,1:rear_locality_input) = test_input(1:end, rear_overlap_input);
        
        % test_in(:,n_kind_input*locality+1:n_kind_input*locality+num_reservoirs_per_worker*chunk_size_input) = test_input(1:end, chunk_begin_input:chunk_end_input);
        test_in(:,rear_locality_input+1:rear_locality_input+num_reservoirs_per_worker*chunk_size_input) = test_input(1:end, chunk_begin_input:chunk_end_input);
        
        % test_in(:,n_kind_input*locality+num_reservoirs_per_worker*chunk_size_input+1:2*n_kind_input*locality+num_reservoirs_per_worker*chunk_size_input) = test_input(1:end, forward_overlap_input);
        test_in(:,rear_locality_input+num_reservoirs_per_worker*chunk_size_input+1:rear_locality_input+num_reservoirs_per_worker*chunk_size_input+forward_locality_input) = test_input(1:end, forward_overlap_input);
        
        test_in = sigma*test_in;
        
        test_uu = tf.test_input_sequence;
        % clear tf;
        
        test_u = zeros(test_steps, num_reservoirs_per_worker*chunk_size_data + overlap_size_data); % this will be populated by the input data to the reservoir
        
        % test_u(:,1:n_kind_data*locality) = test_uu(1:end, rear_overlap_data);
        test_u(:,1:rear_locality_data) = test_uu(1:end, rear_overlap_data);
        
        % test_u(:,n_kind_data*locality+1:n_kind_data*locality+num_reservoirs_per_worker*chunk_size_data) = test_uu(1:end, chunk_begin_data:chunk_end_data);
        test_u(:,rear_locality_data+1:rear_locality_data+num_reservoirs_per_worker*chunk_size_data) = test_uu(1:end, chunk_begin_data:chunk_end_data);
        
        % test_u(:,n_kind_data*locality+num_reservoirs_per_worker*chunk_size_data+1:2*n_kind_data*locality+num_reservoirs_per_worker*chunk_size_data) = test_uu(1:end,forward_overlap_data);
        test_u(:,rear_locality_data+num_reservoirs_per_worker*chunk_size_data+1:rear_locality_data+num_reservoirs_per_worker*chunk_size_data+forward_locality_data) = test_uu(1:end,forward_overlap_data);
        
        test_u = sigma*test_u;
        
        tf = [];
        test_uu = [];
        % fprintf('start res train predict');
        % [pred_collect, RMSE] = res_predict(x, w_out, w, w_in, transpose(test_u), resparams, jobid, locality, chunk_size, pred_marker_array, sync_length);
        
        % fprintf('start learning %s\n', learn);
        switch learn
            case 'LSM_GD_short_prediction_time'
                locality = locality_array;
                % max_iter = 20;
                max_iter = 10;
                % max_iter = 5;
                % max_iter = 2;
                % E_list = zeros(1, max_iter);
                T_list = zeros(1, max_iter);
                l_list = zeros(1, max_iter);
                iter = 1;

                while iter <= max_iter
                    switch l 
                        case 1
                            fprintf('\n------------------------------\n');  
                            fprintf('iter%d->\n', iter);
                    end
                    l_list(iter) = locality;
                    currentAndNextT = zeros(2, 1);
                    for dlocality = 1:2
                    % for dlocality = 2:-1:1
                        locality = locality + (dlocality-1);
                        num_inputs2 = chunk_size_data + 2 * (n_kind_data + n_kind_input) * locality;
                        switch l
                            case 1
                                fprintf('locality: %d\n', locality);
                                rng(jobid+iter*2+dlocality);
                                % [num_inputs2,~] = size(u.');
                                % A = generate_reservoir(resparams.N, resparams.radius, resparams.degree, labindex, jobid);
                                switch reservoir_kind
                                    case 'uniform'
                                        A = generate_reservoir(resparams.N, resparams.radius, resparams.degree, labindex, jobid);
                                    case 'spatial'
                                        loc = int32(num_inputs2/2/nodes_per_input/4);
                                        A = generate_spatial_reservoir(resparams.N, resparams.radius, resparams.degree, labindex, jobid, nodes_per_input, loc);
                                end
                                q = resparams.N/num_inputs2;
                                
                                win = zeros(resparams.N, num_inputs2);
                                % display(size(win));
                                n_additional = rem(resparams.N, num_inputs2);
                                nodes_per_in = floor(double(resparams.N) / double(num_inputs2));
                                % display('eee');
                                add_id = randsample(num_inputs2, n_additional);
                                % display('eeeq');
                                nodes_list = double(nodes_per_in) * ones(1, num_inputs2);
                                % display('fff');
                                nodes_list(add_id) = nodes_list(add_id) + 1;
                                beg = 1;
                                for i=1:num_inputs2
                                    rng(i)
                                    % ip = (-1 + 2*rand(q,1));
                                    % win((i-1)*q+1:i*q,i) = ip;
                                    ip = (-1 + 2*rand(nodes_list(i), 1));
                                    fin = beg + nodes_list(i)-1;
                                    win(beg:fin,i) = ip;
                                    beg = fin + 1;
                                end
                                % display('finish defining reservoirs');
                                % display(size(win));
                                % display(size(A));

                                % data = transpose(u);
                                data = transpose(u(:, 1:locality*2+chunk_size_data));
                                % fprintf('size of data: %d, %d', size(data));
                                
                                % wout = zeros(chunk_size, resparams.N);
                                % [x, wout] = recursive_least_square(resparams, u.', win, A, wout, locality, chunk_size, sync_length);
                                states = reservoir_layer(A, win, data, resparams, train_in);
                                % states(2:2:resparams.N,:) = states(2:2:resparams.N,:).^2;
                                
                                wout = fit(resparams, states, data(locality+1:locality+chunk_size_data,resparams.discard_length + 1:resparams.discard_length + resparams.train_length));
                                x = states(:,end);
                                
                                % error = wout*states - data(locality+1:locality+chunk_size,resparams.discard_length + 1:resparams.discard_length + resparams.train_length);
                                % error = error .^ 2;
                                % RMSE = sqrt(mean(mean(error)));
                                
                                destinations = 2:num_workers;
                                % fprintf('started sending weights\n');
                                spmdBarrier;
                                labSend(A, destinations);
                                labBarrier;
                                labSend(win, destinations);
                                labBarrier;
                                labSend(wout, destinations);
                                labBarrier;
                                labSend(x, destinations);
                                % fprintf('finished sending weights\n');
                            otherwise
                                u = [];
                                % fprintf('started receiving weights\n');
                                labBarrier;
                                A = labReceive();
                                labBarrier;
                                win = labReceive();
                                labBarrier;
                                wout = labReceive();
                                labBarrier;
                                x = labReceive();
                                % fprintf('finished receiving weights\n');
                        end
                        test_data = transpose(test_u(:, locality+1:locality+chunk_size_data));
                        test_in = transpose(test_in);
                        pred_collect = res_predict_GD(x, wout, A, win, transpose(test_u(:, 1:2*locality+chunk_size_data)), resparams, jobid, locality, n_kind_data, chunk_size_data, num_reservoirs_per_worker, pred_marker_array, sync_length, transpose(test_in));
                        % fprintf('calculated pred_collect of %d\n', l);
                        collated_prediction = gcat(pred_collect,1,1);
                        % prediction error
                        % val_length = ;
                        num_preds = length(pred_marker_array);
                        trajectories_true = zeros(chunk_size_data, num_preds*resparams.predict_length);
                        diff = zeros(chunk_size_data, num_preds*resparams.predict_length);
                        % display('eee');
                        for pred_iter = 1:num_preds
                            prediction_marker = pred_marker_array(pred_iter);
                            % trajectories_true(:, (pred_iter-1)*resparams.predict_length + 1: pred_iter*resparams.predict_length) = transpose(sigma*test_file.test_input_sequence(prediction_marker+sync_length + 1:prediction_marker+sync_length + resparams.predict_length,:));
                            % display(pred_iter);
                            trajectories_true(:, (pred_iter-1)*resparams.predict_length + 1: pred_iter*resparams.predict_length) = sigma * test_data(:, prediction_marker+sync_length + 1:prediction_marker+sync_length + resparams.predict_length);
                            % diff(:, (pred_iter-1)*resparams.predict_length+1:pred_iter*resparams.predict_length) ...
                            %     = transpose(sigma*test_file.test_input_sequence(prediction_marker+sync_length +1:prediction_marker + sync_length + resparams.predict_length,:))...
                            % -  pred_collect(:,(pred_iter-1)*resparams.predict_length+1:pred_iter*resparams.predict_length);
                            % display('ggg');
                            diff(:, (pred_iter-1)*resparams.predict_length+1:pred_iter*resparams.predict_length) ...
                                = test_data(:, prediction_marker+sync_length +1:prediction_marker + sync_length + resparams.predict_length) ...
                            -  pred_collect(:,(pred_iter-1)*resparams.predict_length+1:pred_iter*resparams.predict_length);
                        end
                        % display('hhh');
                        RMSE = sqrt(mean(diff.^2, 1));
                        % test_data = transpose(test_u(:, locality+1:locality+chunk_size));
                        % error = pred_collect - test_data(locality+1:locality+chunk_size, sync_length+1:sync_length+resparams.predict_length);
                        % RMSE = sum(error, 1);
                        % display('ii');
                        mean_error = zeros(size(RMSE(1:resparams.predict_length)));

                        % threshold = 0.15;
                        threshold = 0.3;
                        % threshold = 1.0;
                        for k = 1:num_preds
                            mean_error = mean_error + RMSE(resparams.predict_length*(k-1)+1:resparams.predict_length*k);
                            out_times = find(RMSE(resparams.predict_length*(k-1)+1:resparams.predict_length*k) > threshold);
                            out_time = out_times(1);
                            currentAndNextT(dlocality) = currentAndNextT(dlocality) + out_time;
                        end
                        % display('kk');
                        mean_error = mean_error / num_preds;
                        out_time2 = find(mean_error>threshold);
                        currentAndNextT(dlocality) = currentAndNextT(dlocality) / num_preds;
                        % display(locality);
                        % display(currentAndNextT);
                    end
                    locality = locality - (dlocality-1);
                    currentAndNextT = gcat(currentAndNextT, 1); 
                    T_list(iter) = mean(currentAndNextT(1, :), 1); 
                    % T_list(iter) = mean(currentAndNextT(1:2:end), 1); 
                    % nextT = mean(currentAndNextT(2:2:end), 1); 
                    nextT = mean(currentAndNextT(2), 1); 
                    % currentT = mean(currentAndNextT(1:2:end), 1); 
                    currentT = mean(currentAndNextT(1), 1); 
                    gradT = nextT - currentT;
                    switch l
                        case 1
                            fprintf('\n------------------------------\n');
                            fprintf('iter%d->\n', iter); 
                                    
                            % fprintf('  locality:%d\n', locality-1);
                            fprintf('  T(%d)=%f\n  T(%d)=%f\n', locality+1, nextT, locality, currentT);
                            fprintf('  gradE: %f\n', gradT);
                    end

                    % delta_locality = int32(150/(5+5*iter)); N = 840;
                    delta_locality = 1;
                    % delta_locality = int32(50/(5+5*iter)); % N = 64;
                    % delta_locality = int32(15/(4+1*iter)); % N = 64;
                    % delta_locality = 20;
                    % display(delta_locality);
                    % display(10/iter);
                    if gradT < 0
                        locality = locality - min(locality-1, delta_locality);
                        % fprintf('decrease locality to %d\n', locality);
                    else
                        locality = locality + delta_locality;
                        % fprintf('increase locality to %d\n', locality);
                    end
                    % switch l
                    %     case 1
                    %         labBarrier;
                    %         labSend(locality, 2:num_workers);
                    %     otherwise
                    %         labBarrier;
                    %         labSend(RMSE, 1);
                    %         labBarrier;
                    %         locality = labReceive(1);
                    % end
                    % E_list(iter) = RMSE;
                    iter = iter + 1;
                    collated_l = gcat(l_list, 1, 1);
                    collated_T = gcat(T_list, 1, 1);
                    bef = size(trajectories_true);
                    trajectories_true = gcat(trajectories_true, 1);
                    aft = size(trajectories_true);
                    diff = gcat(diff, 1);
                    % filename = [data_dir '/', data_kind, '/', data_kind, '_result_' learn '_uniform_train', num2str(train_steps), '_node', num2str(approx_reservoir_size) '-L' num2str(L) '-radius' num2str(rho) '-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter', num2str(which_index_iter) '.mat'];
                    % filename = [data_dir '/', data_kind, '/', data_kind, 'result_linear_train', num2str(train_steps), '_node', num2str(approx_reservoir_size) '-L' num2str(L) '-radius' num2str(rho) '-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter', num2str(which_index_iter) '.mat'];
                    % filename = [data_dir '/', data_kind, '/', data_kind '100-' num2str(approx_reservoir_size) '-L' num2str(L) '-radius' num2str(rho) '-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter', num2str(which_index_iter) '.mat'];
                    % save(filename, 'pred_collect', 'error', 'diff', 'resparams', 'RMSE_mean', 'pred_marker_array', 'trajectories_true', 'locality', 'chunk_size', 'runtime', '-v7.3');
                    % dsave filename pred_collect error diff resparams
                    % display(filename);
                end

            % case 'LSM_GD_training_error'
            %     % max_iter = 20;
            %     max_iter = 5;
            %     iter = 1;
            %     locality = locality_array;
            %     E_list = zeros(1, max_iter);
            %     l_list = zeros(1, max_iter);
            %     while iter <= max_iter
            %         % reservoir
            %         locality = locality + 1 * rem(l, 2); % 奇数のワーカーはlocalityが一つ大きなリザバーで学習
            % 
            %         l_list(iter) = locality;
            %         num_inputs2 = chunk_size + 2 * locality;
            %         A = generate_reservoir(resparams.N, resparams.radius, resparams.degree, labindex, jobid+iter+l);
            %         q = resparams.N/num_inputs2;
            %         win = zeros(resparams.N, num_inputs2);
            %         % display(size(win));
            %         n_additional = rem(resparams.N, num_inputs2);
            %         nodes_per_in = floor(double(resparams.N) / double(num_inputs2));
            %         % display('eee');
            %         add_id = randsample(num_inputs2, n_additional);
            %         % display('eeeq');
            %         nodes_list = double(nodes_per_in) * ones(1, num_inputs2);
            %         % display('fff');
            %         nodes_list(add_id) = nodes_list(add_id) + 1;
            %         beg = 1;
            %         for i=1:num_inputs2
            %             rng(i)
            %             % ip = (-1 + 2*rand(q,1));
            %             % win((i-1)*q+1:i*q,i) = ip;
            %             ip = (-1 + 2*rand(nodes_list(i), 1));
            %             fin = beg + nodes_list(i)-1;
            %             win(beg:fin,i) = ip;
            %             beg = fin + 1;
            %         end
            %         % display('finish defining reservoirs');
            %         % display(size(win));
            %         % display(size(A));
            % 
            %         data = transpose(u(:, 1:locality*2+chunk_size));
            %         % fprintf('size of data: %d, %d', size(data));
            % 
            %         states = reservoir_layer(A, win, data, resparams);
            %         % states(2:2:resparams.N,:) = states(2:2:resparams.N,:).^2;
            % 
            %         wout = fit(resparams, states, data(locality+1:locality+chunk_size,resparams.discard_length + 1:resparams.discard_length + resparams.train_length));
            %         x = states(:,end);
            % 
            %         % test_data = transpose(test_u(:, 1:locality*2+chunk_size));
            %         % pred_collect = res_predict(x, wout, A, win, test_data, resparams, jobid, locality, chunk_size, pred_marker_array, sync_length);
            % 
            %         % prediction error
            %         % error = pred_collect - test_data(locality+1:locality+chunk_size, sync_length+1:sync_length+resparams.predict_length);
            %         % train error
            %         error = wout*states - data(locality+1:locality+chunk_size,resparams.discard_length + 1:resparams.discard_length + resparams.train_length);
            %         RMSE = sqrt(mean(mean(error.^2)));
            %         RMSEs = zeros(num_workers, 1);
            %         switch l
            %             case 1
            %                 fprintf('\n------------------------------\n');
            %                 fprintf('iter%d->\n', iter);
            %                 RMSEs(1) = RMSE;
            %                 labBarrier;
            %                 for lab = 2:num_workers
            %                     RMSEs(lab) = labReceive(lab);
            %                 end
            %                 % display(RMSEs);
            %                 E_locality_delta = mean(RMSEs(1:2:end));
            %                 E_locality = mean(RMSEs(2:2:end));
            %                 delta_E =  E_locality_delta - E_locality;
            %                 gradE = delta_E / 1;
            % 
            %                 fprintf('  locality:%d\n', locality-1);
            %                 fprintf('  E(%d)=%f\n  E(%d)=%f\n', locality, E_locality_delta, locality-1, E_locality);
            %                 fprintf('  gradE: %f\n', gradE);
            % 
            %                 % delta_locality = int32(150/(5+5*iter)); N = 840;
            %                 delta_locality = int32(50/(5+5*iter)); % N = 64;
            %                 % display(delta_locality);
            %                 % display(10/iter);
            %                 if gradE > 0
            %                     locality = locality - min(locality-1, delta_locality);
            %                     fprintf('decrease locality to %d\n', locality);
            %                 else
            %                     locality = locality + delta_locality;
            %                     fprintf('increase locality to %d\n', locality);
            %                 end
            %                 labBarrier;
            %                 labSend(locality, 2:num_workers);
            %             otherwise
            %                 labBarrier;
            %                 labSend(RMSE, 1);
            %                 labBarrier;
            %                 locality = labReceive(1);
            %         end
            %         E_list(iter) = RMSE;
            %         iter = iter + 1;
            %     end
            %     % exit;
            %     % pred_collect = 0;
            %     collated_prediction = 0;
            %     collated_l = gcat(l_list, 1, 1);
            %     collated_E = gcat(E_list, 1, 1);
            
            case 'LSM_common'
                % num_inputs2 = chunk_size + 2 * locality;
                % num_inputs2 = chunk_size_data + chunk_size_input + 2 * (n_kind_data + n_kind_input) * locality; % size(train_in, 2);
                num_inputs2 = chunk_size_data + chunk_size_input + (rear_locality_data+rear_locality_input+forward_locality_data+forward_locality_input); % size(train_in, 2);
                switch l
                    case 1
                        rng(jobid);
                        % [num_inputs2,~] = size(u.');
                        % A = generate_reservoir(resparams.N, resparams.radius, resparams.degree, labindex, jobid);
                        % loc = int32(num_inputs2 / 2) - 1;
                        % loc = int32(num_inputs2/2/nodes_per_input) - 1;
                        % loc = int32(num_inputs2/2/nodes_per_input/4);
                        % A = generate_spatial_reservoir(resparams.N, resparams.radius, resparams.degree, labindex, jobid, nodes_per_input, loc);
                        
                        switch reservoir_kind
                            case 'uniform'
                                A = generate_reservoir(resparams.N, resparams.radius, resparams.degree, labindex, jobid);
                            case 'spatial'
                                % loc = 1;
                                % loc = int32(num_inputs2/2/4);
                                % loc = int32(num_inputs2/2/nodes_per_input/4);
                                A = generate_spatial_reservoir(resparams.N, resparams.radius, resparams.degree, labindex, jobid, nodes_per_input, width);
                        end

                        q = resparams.N/num_inputs2;
                        
                        win = zeros(resparams.N, num_inputs2);
                        % display(size(win));
                        n_additional = rem(resparams.N, num_inputs2);
                        nodes_per_in = floor(double(resparams.N) / double(num_inputs2));
                        % display('eee');
                        add_id = randsample(num_inputs2, n_additional);
                        % display('eeeq');
                        nodes_list = double(nodes_per_in) * ones(1, num_inputs2);
                        % display('fff');
                        nodes_list(add_id) = nodes_list(add_id) + 1;
                        beg = 1;
                        for i=1:num_inputs2
                            rng(i)
                            % ip = (-1 + 2*rand(q,1));
                            % win((i-1)*q+1:i*q,i) = ip;
                            ip = (-1 + 2*rand(nodes_list(i), 1));
                            fin = beg + nodes_list(i)-1;
                            win(beg:fin,i) = ip;
                            beg = fin + 1;
                        end
                        win = sparse(win);
                        % display('w_in');

                        % [num_inputs2,~] = size(u.');
                        % A = generate_reservoir(resparams.N, resparams.radius, resparams.degree, labindex, jobid);
                        % q = resparams.N/num_inputs2;
                        % win = zeros(resparams.N, num_inputs2);
                        % for i=1:num_inputs2
                        %     rng(i)
                        %     ip = (-1 + 2*rand(q,1));
                        %     win((i-1)*q+1:i*q,i) = ip;
                        % end
                        data = transpose(u);
                        % wout = zeros(chunk_size, resparams.N);
                        % [x, wout] = recursive_least_square(resparams, u.', win, A, wout, locality, chunk_size, sync_length);
                        % states = reservoir_layer(A, win, data(1:chunk_size_data+2*n_kind_data*locality, :), resparams, transpose(train_in(:, 1:chunk_size_input+2*num_inputs_input*locality)));
                        states = reservoir_layer(A, win, data(1:chunk_size_data+(rear_locality_data+forward_locality_data), :), resparams, transpose(train_in(:, 1:chunk_size_input+(rear_locality_input+forward_locality_input))));
                        % states(2:2:resparams.N,:) = states(2:2:resparams.N,:).^2;
                        % display('reservoir layer');
                        
                        % wout = fit(resparams, states, data(n_kind_data*locality+1:n_kind_data*locality+chunk_size_data,resparams.discard_length + 1:resparams.discard_length + resparams.train_length));
                        wout = fit(resparams, states, data(rear_locality_data+1:rear_locality_data+chunk_size_data,resparams.discard_length + 1:resparams.discard_length + resparams.train_length));
                        x = states(:,end);
                        % display('fit');

                        % error = wout*states - data(n_kind_data*locality+1:n_kind_data*locality+chunk_size_data,resparams.discard_length + 1:resparams.discard_length + resparams.train_length);
                        error = wout*states - data(rear_locality_data+1:rear_locality_data+chunk_size_data,resparams.discard_length + 1:resparams.discard_length + resparams.train_length);
                        error = error .^ 2;
                        RMSE = sqrt(mean(mean(error)));
                        
                        destinations = 2:num_workers;
                        % fprintf('started sending weights\n');
                        labBarrier;
                        labSend(A, destinations);
                        labBarrier;
                        labSend(win, destinations);
                        labBarrier;
                        labSend(wout, destinations);
                        labBarrier;
                        labSend(x, destinations);
                        % fprintf('finished sending weights\n');
                    otherwise
                        u = [];
                        % fprintf('started receiving weights\n');
                        labBarrier;
                        A = labReceive();
                        labBarrier;
                        win = labReceive();
                        labBarrier;
                        wout = labReceive();
                        labBarrier;
                        x = labReceive();
                        % fprintf('finished receiving weights\n');
                end
                A_concat = zeros(resparams.N*num_reservoirs_per_worker, resparams.N*num_reservoirs_per_worker);
                % num_inputs_concat = (n_kind_data+n_kind_input)*locality*2+num_reservoirs_per_worker*(chunk_size_input+chunk_size_data);
                num_inputs_concat = (forward_locality_data+rear_locality_data+forward_locality_input+rear_locality_input)+num_reservoirs_per_worker*(chunk_size_input+chunk_size_data);
                win_concat = zeros(resparams.N*num_reservoirs_per_worker, num_inputs_concat);
                wout_concat = zeros((chunk_size_data+chunk_size_input)*num_reservoirs_per_worker, resparams.N*num_reservoirs_per_worker);
                for k = 1:num_reservoirs_per_worker
                    row_index = resparams.N*(k-1)+1:resparams.N*k;
                    col_index = resparams.N*(k-1)+1:resparams.N*k;
                    % col_index = mod((resparams.N*(k-1)+1:resparams.N*k)-1, resparams.N*num_reservoirs_per_worker) + 1;
                    A_concat(row_index, col_index) = A;

                    row_index = resparams.N*(k-1)+1:resparams.N*k;
                    % col_index = (chunk_size_input+chunk_size_data)*(k-1)+1:(chunk_size_input+chunk_size_data)*k+2*locality*(n_kind_input+n_kind_data);
                    col_index = (chunk_size_input+chunk_size_data)*(k-1)+1:(chunk_size_input+chunk_size_data)*k+(forward_locality_data+rear_locality_data+forward_locality_input+rear_locality_input);
                    win_concat(row_index, col_index) = win;
                    
                    row_index = (chunk_size_input+chunk_size_data)*(k-1)+1:(chunk_size_input+chunk_size_data)*k;
                    col_index = resparams.N*(k-1)+1:resparams.N*k;
                    wout_concat(row_index, col_index) = wout;
                end
                A_concat = sparse(A_concat); 
                win_concat = sparse(win_concat); 
                
                pred_collect = res_predict(repmat(x, num_reservoirs_per_worker, 1), wout_concat, A_concat, win_concat, transpose(test_u), resparams, jobid, rear_locality_data, forward_locality_data, n_kind_data, chunk_size_data, num_reservoirs_per_worker, pred_marker_array, sync_length, transpose(test_in));
                % pred_collect = res_predict(repmat(x, num_reservoirs_per_worker, 1), wout_concat, A_concat, win_concat, transpose(test_u), resparams, jobid, locality, n_kind_data, chunk_size_data, num_reservoirs_per_worker, pred_marker_array, sync_length, transpose(test_in));
                % pred_collect = res_predict(x, wout, A, win, transpose(test_u), resparams, jobid, locality, n_kind_data, chunk_size_data, pred_marker_array, sync_length, transpose(test_in));
                % fprintf('calculated pred_collect of %d\n', l);
                collated_prediction = gcat(pred_collect,1,1);
            case 'LSM'
                if iter
                    [pred_collect, RMSE] = res_train_predict_iter(transpose(u), transpose(test_u), resparams, jobid, locality, n_kind_data, chunk_size_data, pred_marker_array, sync_length, transpose(train_in), transpose(test_in));
                else
                    % [pred_collect, RMSE] = res_train_predict(transpose(u), transpose(u), resparams, jobid, locality, n_kind_data, chunk_size_data, pred_marker_array, sync_length, nodes_per_input, transpose(train_in), transpose(train_in));
                    % [pred_collect, RMSE] = res_train_predict(transpose(u), transpose(test_u), resparams, jobid, locality, n_kind_data, chunk_size_data, num_reservoirs_per_worker, pred_marker_array, sync_length, nodes_per_input, transpose(train_in), transpose(test_in));
                    [pred_collect, RMSE] = res_train_predict(transpose(u), transpose(test_u), resparams, jobid, rear_locality_data, forward_locality_data, n_kind_data, chunk_size_data, num_reservoirs_per_worker, pred_marker_array, sync_length, nodes_per_input, transpose(train_in), transpose(test_in));
                end
                % fprintf('calculated pred_collect of %d\n', l);
                collated_prediction = gcat(pred_collect,1,1);
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
                
                wout = zeros(chunk_size_data, resparams.N);
                [x, wout] = recursive_least_square(resparams, u.', win, A, wout, locality, chunk_size_data, sync_length);
                pred_collect = res_predict(x, wout, A, win, transpose(test_u), resparams, jobid, locality, chunk_size_data, pred_marker_array, sync_length);
                % fprintf('calculated pred_collect of %d\n', l);
                collated_prediction = gcat(pred_collect,1,1);
            otherwise 
                fprintf('error! learning method is invalid!\n');
        end
    end
    fprintf('prediction complete\n');
    runtime = toc;
    % fprintf('finished prediction\n');
    approx_reservoir_size = approx_reservoir_size{1};
    locality = locality{1};
    num_workers = num_workers{1};
    learn = learn{1};
    % jobid = jobid{1};
    % data_file = m{1};
    % test_file = tf{1};
    data_kind = data_kind{1}; reservoir_kind = reservoir_kind{1};
    max_lyapunov = max_lyapunov{1};
    train_steps = train_steps{1}; test_steps = test_steps{1};
    L = L{1};
    switch data_kind
        case 'LCD'
            dt = 5e-4;
            N = N{1};
            test_filename = test_filename{1};
            test_file = matfile(test_filename);
        case 'CGL' 
            dt = 0.07;
            N = N{1}; c1 = c1{1}; c2 = c2{1};
            test_filename = test_filename{1};
            % data_file = load([data_dir 'CGL_L18_N_32_dps80000.mat']); % CGL
            test_file = load(test_filename); % CGL
            % test_file = load([data_dir 'CGL_L' num2str(L) '_N_' num2str(N) '_dps' num2str(test_steps) '.mat']); % CGL
        case 'KS'
            dt = 1/4;
            N = N{1}; 
            % data_file = load([data_dir 'train_input_sequence.mat']); % KS
            % test_file = load([data_dir 'test_input_sequence.mat']); % KS
            test_file = matfile([data_dir data_kind '_L', num2str(L) '_N_', num2str(N) '_dps', num2str(test_steps) '.mat']); % KS
        case 'KS_slow'
            dt = 1/8;
            N = N{1}; 
            % data_file = load([data_dir 'train_input_sequence.mat']); % KS
            % test_file = load([data_dir 'test_input_sequence.mat']); % KS
            test_file = matfile([data_dir data_kind '_L', num2str(L) '_N_', num2str(N) '_dps', num2str(test_steps) '.mat']); % KS
        case 'KS_slow_short'
            dt = 1/8;
            N = N{1}; 
            % data_file = load([data_dir 'train_input_sequence.mat']); % KS
            % test_file = load([data_dir 'test_input_sequence.mat']); % KS
            test_file = matfile([data_dir data_kind '_L', num2str(L) '_N_', num2str(N) '_dps', num2str(test_steps) '.mat']); % KS
        case 'KS_sync'
            dt = 1/4;
            N = N{1}; 
            % data_file = load([data_dir 'train_input_sequence.mat']); % KS
            % test_file = load([data_dir 'test_input_sequence.mat']); % KS
            test_file = matfile([data_dir data_kind '_L', num2str(L) '_N_', num2str(N) '_dps', num2str(test_steps) '.mat']); % KS
            % test_file = matfile([data_dir data_kind '_L', num2str(L) '_N_', num2str(N) '_dps', num2str(train_steps) '.mat']); % KS
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
    chunk_size_data = chunk_size_data{1};
    num_inputs_data = num_inputs_data{1};
    pred_collect = collated_prediction{1};
    pred_marker_array = pred_marker_array{1};
    num_preds = length(pred_marker_array);
    
    switch learn
        case 'LSM_GD_short_prediction_time'
            RMSE_mean = 0;
            % y = collated_E{1};
            y = collated_T{1};
            figure(); errorbar(mean(y, 1), std(y, 1));
            xlabel('iterations'); ylabel('time'); grid on;
            figure(); plot(collated_l{1}(1, :));
            xlabel('iterations'); ylabel('locality'); grid on;
        case 'LSM_GD'
            % y = collated_E{1};
            y = collated_E{1}(1:2:end);
            figure(); errorbar(mean(y, 1), std(y, 1));
            xlabel('iterations'); ylabel('RMSE'); grid on;
            figure(); plot(collated_l{1}(1, :));
            xlabel('iterations'); ylabel('locality'); grid on;
        case 'LSM'
            RMSE_mean = 0;
            for i = 1:num_workers
                RMSE_mean = RMSE_mean + RMSE{i};
                i;
            end
            RMSE_mean = RMSE_mean / num_workers;
        case 'RLS'
            RMSE_mean = 0;
        case 'LSM_common'
            RMSE_mean = RMSE{1};
    end
    switch learn
        case 'LSM_GD_short_prediction_time'
            trajectories_true = trajectories_true{1};
            diff = diff{1};
            error = RMSE{1};
        otherwise
            diff = zeros(num_inputs_data, num_preds*resparams.predict_length);
            trajectories_true = zeros(num_inputs_data, num_preds*resparams.predict_length);
            for pred_iter = 1:num_preds
                prediction_marker = pred_marker_array(pred_iter);
                trajectories_true(:, (pred_iter-1)*resparams.predict_length + 1: pred_iter*resparams.predict_length) = transpose(sigma*test_file.test_input_sequence(prediction_marker+sync_length + 1:prediction_marker+sync_length + resparams.predict_length,:));
                diff(:, (pred_iter-1)*resparams.predict_length+1:pred_iter*resparams.predict_length) ...
                    = transpose(sigma*test_file.test_input_sequence(prediction_marker+sync_length +1:prediction_marker + sync_length + resparams.predict_length,:))...
                -  pred_collect(:,(pred_iter-1)*resparams.predict_length+1:pred_iter*resparams.predict_length);
                error = sqrt(mean(diff.^2, 1));
            end
    end


    which_index_iter = which_index_iter{1};

    %% save results
%    filename = ['KS100-' num2str(approx_reservoir_size) '-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter', num2str(which_index_iter)];
    % filename = ['/lustre/jpathak/KS100/KS100-' num2str(approx_reservoir_size) '-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter', num2str(which_index_iter)];
    % filename = [data_dir '/KS100-' num2str(approx_reservoir_size) '-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter', num2str(which_index_iter)];
    switch reservoir_kind 
        case 'uniform'
            % filename = [data_dir '/', data_kind, '/', data_kind, '_result_' learn '_' reservoir_kind '_reservoir' '_train', num2str(train_steps), '_node', num2str(approx_reservoir_size) '-L' num2str(L) '-radius' num2str(rho) '-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter', num2str(which_index_iter) '.mat'];
            filename = [data_dir '/', data_kind, '/', data_kind, '_result_' learn '_' reservoir_kind '_reservoir' '_train', num2str(train_steps), '_node', num2str(approx_reservoir_size) '-L' num2str(L) '-N' num2str(N) '-radius' num2str(rho) '-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter', num2str(which_index_iter) '.mat'];
        case 'spatial'
            % filename = [data_dir '/', data_kind, '/', data_kind, '_result_' learn '_' reservoir_kind '_reservoir' '_train', num2str(train_steps), '_node', num2str(approx_reservoir_size) '-L' num2str(L) '-radius' num2str(rho) '-width' num2str(width) '-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter', num2str(which_index_iter) '.mat'];
            filename = [data_dir '/', data_kind, '/', data_kind, '_result_' learn '_' reservoir_kind '_reservoir' '_train', num2str(train_steps), '_node', num2str(approx_reservoir_size) '-L' num2str(L) '-N' num2str(N) '-radius' num2str(rho) '-width' num2str(width) '-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter', num2str(which_index_iter) '.mat'];
    end
            % filename = [data_dir '/', data_kind, '/', data_kind, 'result_linear_train', num2str(train_steps), '_node', num2str(approx_reservoir_size) '-L' num2str(L) '-radius' num2str(rho) '-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter', num2str(which_index_iter) '.mat'];
    % filename = [data_dir '/', data_kind, '/', data_kind '100-' num2str(approx_reservoir_size) '-L' num2str(L) '-radius' num2str(rho) '-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter', num2str(which_index_iter) '.mat'];
    
    % if jobid == 1
    if jobid == 0
        % true, predicted, error data
        save(filename, 'pred_collect', 'error', 'diff', 'resparams', 'RMSE_mean', 'pred_marker_array', 'trajectories_true', 'locality', 'chunk_size_data', 'runtime', '-v7.3');
    else
        % just RMSEs
        save(filename, 'error', 'resparams', 'RMSE_mean', 'pred_marker_array', 'locality', 'chunk_size_data', 'runtime', '-v7.3');
    end
    display(filename);
    
    %% show graphs
    n_steps = size(trajectories_true, 2);
    n_data = size(trajectories_true, 1);
    times = repmat(0:dt*max_lyapunov:(n_steps-1)*dt*max_lyapunov, n_data, 1);
    locations = repmat((1:n_data).', 1, n_steps);
    max_value = max(max(trajectories_true)); min_value = min(min(trajectories_true));
    if show_fig
        switch data_kind 
            case 'KS'
                figure(); 
                subplot(3, 1, 1); surf(times, locations, trajectories_true(:,1:n_steps)); view(0, 90); shading interp, axis tight; xlabel('lyapunov time'); ylabel('x'); title('true data'); colorbar; clim([min_value max_value]); xlim([0 50]);
                subplot(3, 1, 2); surf(times, locations, pred_collect(:,1:n_steps)); view(0, 90); shading interp, axis tight; xlabel('lyapunov time'); ylabel('x'); title('predicted data'); colorbar; clim([min_value max_value]); xlim([0 50]);
                subplot(3, 1, 3); surf(times, locations, diff(:,1:n_steps)); view(0, 90); shading interp, axis tight; xlabel('lyapunov time'); ylabel('x'); title('error'); colorbar; clim(2 * [min_value max_value]); xlim([0 50]);
                sgtitle([data_kind ' L:' num2str(L) ' rho: ' num2str(resparams.radius) ', request pool size: ' num2str(request_pool_size) ', locality: ' num2str(locality)]);
                colormap(jet);
            case 'KS_slow'
                figure(); 
                subplot(3, 1, 1); surf(times, locations, trajectories_true(:,1:n_steps)); view(0, 90); shading interp, axis tight; xlabel('lyapunov time'); ylabel('x'); title('true data'); colorbar; clim([min_value max_value]); xlim([0 50]);
                subplot(3, 1, 2); surf(times, locations, pred_collect(:,1:n_steps)); view(0, 90); shading interp, axis tight; xlabel('lyapunov time'); ylabel('x'); title('predicted data'); colorbar; clim([min_value max_value]); xlim([0 50]);
                subplot(3, 1, 3); surf(times, locations, diff(:,1:n_steps)); view(0, 90); shading interp, axis tight; xlabel('lyapunov time'); ylabel('x'); title('error'); colorbar; clim(2 * [min_value max_value]); xlim([0 50]);
                sgtitle([data_kind ' L:' num2str(L) ' rho: ' num2str(resparams.radius) ', request pool size: ' num2str(request_pool_size) ', locality: ' num2str(locality)]);
                colormap(jet);
            case 'LCD'
                figure(); 
                subplot(3, 1, 1); surf(times, locations, trajectories_true(:,1:n_steps)); view(0, 90); shading interp, axis tight; xlabel('lyapunov time'); ylabel('x'); title('true data'); colorbar; clim([min_value max_value]); xlim([0 1.25]);
                subplot(3, 1, 2); surf(times, locations, pred_collect(:,1:n_steps)); view(0, 90); shading interp, axis tight; xlabel('lyapunov time'); ylabel('x'); title('predicted data'); colorbar; clim([min_value max_value]); xlim([0 1.25]);
                subplot(3, 1, 3); surf(times, locations, diff(:,1:n_steps)); view(0, 90); shading interp, axis tight; xlabel('lyapunov time'); ylabel('x'); title('error'); colorbar; clim(2 * [min_value max_value]); xlim([0 1.25]);
                sgtitle([data_kind ' L:' num2str(L) ' rho: ' num2str(resparams.radius) ', request pool size: ' num2str(request_pool_size) ', locality: ' num2str(locality)]);
                colormap(jet);
            case 'KS_sync'
                figure(); 
                subplot(3, 1, 1); surf(times, locations, trajectories_true(:,1:n_steps)); view(0, 90); shading interp, axis tight; xlabel('lyapunov time'); ylabel('x'); title('true data'); colorbar; clim([min_value max_value]); xlim([0 50]);
                subplot(3, 1, 2); surf(times, locations, pred_collect(:,1:n_steps)); view(0, 90); shading interp, axis tight; xlabel('lyapunov time'); ylabel('x'); title('predicted data'); colorbar; clim([min_value max_value]); xlim([0 50]);
                subplot(3, 1, 3); surf(times, locations, diff(:,1:n_steps)); view(0, 90); shading interp, axis tight; xlabel('lyapunov time'); ylabel('x'); title('error'); colorbar; clim(2 * [min_value max_value]); xlim([0 50]);
                sgtitle([data_kind ' L:' num2str(L) ' rho: ' num2str(resparams.radius) ', request pool size: ' num2str(request_pool_size) ', locality: ' num2str(locality)]);
                colormap(jet);
            case 'CGL'
                figure(); 
                subplot(3, 1, 1); surf(times, locations, [trajectories_true(1:2:end-1,1:n_steps); trajectories_true(2:2:end,1:n_steps)]); view(0, 90); shading interp, axis tight; xlabel('lyapunov time'); ylabel('x'); title('true data'); colorbar; clim([min_value max_value]); xlim([0 50]);
                subplot(3, 1, 2); surf(times, locations, [pred_collect(1:2:end-1,:); pred_collect(2:2:end,:)]); view(0, 90); shading interp, axis tight; xlabel('lyapunov time'); ylabel('x'); title('predicted data'); colorbar; clim([min_value max_value]); xlim([0 50]);
                subplot(3, 1, 3); surf(times, locations, [diff(1:2:end-1,:); diff(2:2:end, :)]); view(0, 90); shading interp, axis tight; xlabel('lyapunov time'); ylabel('x'); title('error'); colorbar; clim(2 * [min_value max_value]); xlim([0 50]);
                sgtitle([data_kind ' L:' num2str(L) ' rho: ' num2str(resparams.radius) ', request pool size: ' num2str(request_pool_size) ', locality: ' num2str(locality)]);
                colormap(jet);
        end

        figure(); plot(times(1,:), sqrt(mean(diff.^2, 1)));
        xlabel('time (lyapunov*second'); ylabel('RMSE'); title('error');
    end

    
    % naverage = 1;
    % % naverage = 50;
    % % naverage = 2000;
    % interval = floor(10/dt/max_lyapunov);
    % % interval = 20;
    % start_step = floor(6/dt/max_lyapunov);
    % 
    % % Temporal Powerspectrum
    % % % [true_temporal_p, f] = pspectrum(trajectories_true(1, :), 1/dt);
    % % % [pred_temporal_p, f] = pspectrum(pred_collect(1, :), 1/dt);
    % % % figure();
    % % % plot(f, true_temporal_p, 'DisplayName', ['true']);
    % % % hold off;
    % % % hold on;
    % % % plot(f, pred_temporal_p, 'DisplayName', ['prediction']);
    % % % hold off;
    % % % title(['Temporal Powerspectrum' ' g=' num2str(request_pool_size)]);
    % 
    % true_sum_p = zeros(4096, 1);
    % pred_sum_p = zeros(4096, 1);
    % % true_spatial_ps = zeros(4096, naverage);
    % % pred_spatial_ps = zeros(4096, naverage);
    % figure();
    % for i =start_step:interval:start_step+interval*naverage-1
    %     % [p, f] = pspectrum(test_input_sequence(i,:), N/L*2*pi);
    %     [true_spatial_p, f] = pspectrum(trajectories_true(:, i), N/L);
    %     [pred_spatial_p, f] = pspectrum(pred_collect(:, i), N/L);
    %     % p = pspectrum(test_input_sequence(i,:), 128/50);
    %     % pp = 20*log10(p);
    %     % true_spatial_ps(:, i) = true_spatial_p;
    %     % pred_spatial_ps(:, i) = pred_spatial_p;
    %     % figure(); plot(f, p);
    %     % set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log');
    %     % xlabel('Spatial Frequency (Hz)'); ylabel('Power Spectrum(dB)');
    %     % fontsize(16, 'points');
    %     true_sum_p = true_sum_p + true_spatial_p;
    %     pred_sum_p = pred_sum_p + pred_spatial_p;
    % end
    % true_sum_p = true_sum_p / naverage;
    % pred_sum_p = pred_sum_p / naverage;
    % % sum_p = sum_p .^ (1/naverage);
    % 
    % hold on;
    % % scatter(2*pi*f, sum_p);
    % plot(2*pi*f, true_sum_p, 'DisplayName', ['true']);
    % hold off;
    % hold on;
    % plot(2*pi*f, pred_sum_p, 'DisplayName', ['pred']);
    % hold off;
    % set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log');
    % % xlabel('Spatial Frequency (Hz)'); 
    % % ylabel('Power Spectrum');
    % fontsize(16, 'points');
    % title(['Spatial Powerspectrum' ' g=' num2str(request_pool_size)]);
    % % title(['average over ' num2str(naverage)]);
    % % title(['Powerspectrum c_1=' num2str(c1) ', c_2=' num2str(c2)]);
    % % title(['Powerspectrum ' 'L=' num2str(L)]);
    % xlim([10^-2 10^1]); ylim([10^-4 10^2]);
    % xticks(logspace(-2, 1, 4)); yticks(logspace(-4, 2, 7));
    % xlabel('q'); ylabel('g_q');
    % legend();
    % grid on;
    % 
    % [max_true_power, max_true_index] = max(true_sum_p);
    % max_true_freq = f(max_true_index);    
    % [max_pred_power, max_pred_index] = max(pred_sum_p);
    % max_pred_freq = f(max_pred_index);
    % fprintf('max spatial frequency for true data: %f\n', max_true_freq);
    % fprintf('max spatial frequency for predicted data: %f\n', max_pred_freq);
    
    %% progress bar
    progress = progress + 1;
    % total = size(request_pool_size) * size(rho_list, 2) * size(locality_list, 2) * size(jobid_list, 2) * size(train_steps_list, 2);
    total = size(width_list, 2) * size(rho_list, 2) * size(locality_list, 2) * size(jobid_list, 2) * size(train_steps_list, 2);
    h = waitbar(progress/total,h,... 
    sprintf('progress: %d/%d', progress, total));
    % if mean(mean(diff.^2, 1)) > 2
    %     close all;
    % end
    % clear pred_marker_array which_index_iter rho_array locality_array;
    toc
    % close all;
    clear trajectories_true pred_collect diff 
end
end
end

end
end
end
% end
if size(request_pool_size_list, 2) > 1
    clearvars -except request_pool_size request_pool_size_list h;
end
% delete(gcp);
end
close(h);
% end




