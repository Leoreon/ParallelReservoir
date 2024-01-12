
% pred_length = 2499; num_pred = 49;
pred_length = 2499; num_pred = 40;
% pred_length = 2499; num_pred = 20;
% pred_length = 2199; num_pred = 20;

%% errors 
% Ntotal = 1680;
% Ntotal = 5040;
% locality = 50;
% num_workers_list = 2:8;

% L = 22; Ntotal = 5040; N=840; locality_list = [7 8 9 10 11 12 20]; num_workers_list = [6]; % jobid_list = 2:3; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% L = 22; Ntotal = 5880; N=840; locality_list = [9 10 11 12]; num_workers_list = [6];  jobid_list = 6:10; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];

%% for gradient descent
% train_steps = 20001; locality_list = 40:130; num_workers_list = [6];  jobid_list = 1; learn = 'LSM_common'; reservoir_kind = 'uniform'; data_kind = 'KS'; L = 22; Ntotal = 5040; N=840;  % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% train_steps = 80000; locality_list = 60:130; num_workers_list = [6];  jobid_list = 1; learn = 'LSM_common'; reservoir_kind = 'uniform'; data_kind = 'KS'; L = 22; Ntotal = 5040; N=840;  % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% train_steps = 80000; locality_list = 60:100; num_workers_list = [6];  jobid_list = [1 11:14]; learn = 'LSM_common'; reservoir_kind = 'uniform'; data_kind = 'KS'; L = 22; Ntotal = 5040; N=840;  % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% learn = 'LSM_common'; reservoir_kind = 'uniform_fix'; train_steps = 80000; locality_list = 60:130; num_workers_list = [6];  jobid_list = 10; data_kind = 'KS'; L = 22; Ntotal = 5040; N=840; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% learn = 'LSM_common'; reservoir_kind = 'uniform_fix'; train_steps = 20001; locality_list = 60:100; num_workers_list = [6];  jobid_list = 11:14; data_kind = 'KS'; L = 22; Ntotal = 5040; N=840; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% data_kind = 'KS'; L = 22; Ntotal = 5040; N=840; train_steps = 10000; locality_list = 40:130; num_workers_list = [6];  jobid_list = 1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% data_kind = 'KS'; L = 22; Ntotal = 5040; N=840; train_steps = 2500; locality_list = 40:130; num_workers_list = [6];  jobid_list = 1:15; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];

%% for Gradient Descent
% data_kind = 'KS'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; locality_list = 40:130; num_workers_list = [6];  jobid_list = 1:4; reservoir_kind = 'uniform'; learn = 'LSM_common';% locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% data_kind = 'KS'; L = 22; Ntotal = 5040; N=840; locality_list = 40:50; num_workers_list = [6];  jobid_list = 1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% data_kind = 'KS'; L = 22; Ntotal = 5040; N=840; train_steps = 2500; locality_list = 40:70; num_workers_list = [6];  jobid_list = 1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% data_kind = 'KS'; L = 22; Ntotal = 5040; N=840; train_steps = 2500; locality_list = 45:55; num_workers_list = [6];  jobid_list = 1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% data_kind = 'KS'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; locality_list = 40:130; num_workers_list = [6];  jobid_list = 1:4; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% data_kind = 'KS'; learn = 'LSM_common'; reservoir_kind = 'uniform_fix'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; locality_list = 40:180; num_workers_list = [6];  jobid_list = 3; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% data_kind = 'KS'; reservoir_kind = 'uniform'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; locality_list = 140:20:230; num_workers_list = [6];  jobid_list = 3; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% data_kind = 'KS'; reservoir_kind = 'uniform'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; locality_list = 40:240; num_workers_list = [6];  jobid_list = 3; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];

% locality_list = 20:20:200; num_workers_list = [2]; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; jobid_list = 1:3; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% locality_list = 20:20:200; num_workers_list = [6]; jobid_list = 1:3; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% locality_list = 60:5:90; num_workers_list = [5]; jobid_list = 1:3; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];

%% Lyapunov
% beta = 1e-4;
% % % locality_list = 0:20:200; num_workers_list = [2:8]; jobid_list = 1:3; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% % locality_list = 60:70; num_workers_list = 6;  jobid_list = 1e5+[1:10]; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; pred_length = 1000-100-1;% locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% locality_list = 20:20:200; num_workers_list = [6]; jobid_list = 1:10; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% locality_list = 60; num_workers_list = [2:8]; jobid_list = 2; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% locality_list = 60; num_workers_list = [6]; jobid_list = 1001:1010; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% locality_list = 20:20:200; num_workers_list = [4:8]; jobid_list = 1:10; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% locality_list = 20:20:120; num_workers_list = [6]; jobid_list = 1:10; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% locality_list = 40:20:100; num_workers_list = [4]; jobid_list = 1:3; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% locality_list = 100; num_workers_list = [4]; jobid_list = 1:3; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];

% beta = 1;
beta = 1e-2;
% locality_list = 61:70; num_workers_list = 6; jobid_list = 1:3; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 20001; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% locality_list = 41:70; num_workers_list = 6; jobid_list = 1; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 20001; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% % % locality_list = 61:76; num_workers_list = 6; jobid_list = 1; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 20001; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% del_locality = 1; locality_list = [1:201]; num_workers_list = 6; jobid_list = 1; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 20001; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% % del_locality = 10; locality_list = [10:64]; num_workers_list = 6; jobid_list = 1; data_kind = 'KS'; reservoir_kind = 'uniform_fix'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 20001; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
del_locality = 1; locality_list = [20:218 220:340]; num_workers_list = 6; jobid_list = 1; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 20001; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
del_locality = 1; locality_list = [1:340]; num_workers_list = 6; jobid_list = 1; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 20001; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% locality_list = 21:41; num_workers_list = 6; jobid_list = 1; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 20001; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% del_locality = 1;

% locality_list = 0:20:200; num_workers_list = [2:8]; jobid_list = 1:3; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% locality_list = 0:20:200; num_workers_list = [16:-2:4]; jobid_list = 1:3; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 44; Ntotal = 10080; N=1680; train_steps = 80000; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% locality_list = 0:20:20; num_workers_list = [16]; jobid_list = 1:3; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 44; Ntotal = 10080; N=1680; train_steps = 80000; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% locality_list = 60:120; num_workers_list = 6; jobid_list = 1e5+[1:10]; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; pred_length = 2000-100-1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% locality_list = 60:97; num_workers_list = 6; jobid_list = 1e5+[1:10]; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; pred_length = 2000-100-1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% locality_list = 60:70; num_workers_list = 6; jobid_list = 1e5+[1:40]; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; pred_length = 2000-100-1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% locality_list = 60:97; num_workers_list = 6; jobid_list = 1:3; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 20001; pred_length = 3000-500-1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% locality_list = 60:97; num_workers_list = 6; jobid_list = 1:3; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 40000; pred_length = 3000-500-1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% % locality_list = 60:80; num_workers_list = 6; jobid_list = 1e5+[1:30]; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 20001; pred_length = 3000-500-1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% locality_list = 60:2:100; num_workers_list = 6; jobid_list = 1e5+[1:10]; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; pred_length = 2000-100-1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];

% num_workers_list = [6]; locality_list = 0:120; data_kind = 'KS';
% reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L =
% 22; Ntotal = 5040; N=840; train_steps = 80000; jobid_list = 5; %
% locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6
% 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; locality_list = 50:90; num_workers_list = [6];  jobid_list = 1:3; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% num_workers_list = [7]; % data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; locality_list = 40:60;  jobid_list = [1:3]; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% num_workers_list = [8]; data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; locality_list = 80:90;  jobid_list = 1:3; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; locality_list = 41:100; num_workers_list = [8];  jobid_list = 5; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% data_kind = 'KS'; reservoir_kind = 'uniform'; learn = 'LSM_GD_short_prediction_time'; L = 22; Ntotal = 5040; N=840; train_steps = 80000; locality_list = 60:20:200; num_workers_list = [8];  jobid_list = 5; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];


%% 
% data_kind = 'KS_slow'; L = 44; Ntotal = 5040; N=1680; train_steps = 160000; locality_list = 20:20:280; num_workers_list = 2*[2 3 4 5 6 7 8]; jobid_list = 1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
%% 

%% 2章 予測時間と訓練誤差
% data_kind = 'KS'; L = 22; dt = 1/4; Ntotal = 5040; N=840; train_steps = 80000; locality_list = 20:20:160; num_workers_list = [2 3 4 5 6 7 8]; jobid_list = 1:3; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% data_kind = 'KS'; L = 44; dt = 1/4; Ntotal = 10080; N=1680; train_steps = 80000; locality_list = 20:20:280; num_workers_list = 4:2:16; jobid_list = 1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% data_kind = 'KS_slow'; dt = 1/8; L = 22; Ntotal = 5040; N=840; train_steps = 160000; locality_list = 20:20:280; num_workers_list = [2 3 4 5 6 7 8]; jobid_list = 1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% data_kind = 'KS_slow_short'; dt = 1/8; L = 22; Ntotal = 5040; N=840; train_steps = 80000; locality_list = 20:20:280; num_workers_list = [2 3 4 5 6 7 8]; jobid_list = 1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% data_kind = 'KS_slow_short'; dt = 1/8; L = 22; Ntotal = 5040; N=840; train_steps = 80000; locality_list = 20:20:280; num_workers_list = [2 3 4 5 6 7 8]; jobid_list = 1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];
% data_kind = 'KS'; L = 22; dt = 1/4; Ntotal = 5040; N=840; train_steps = 80000; locality_list = 20:20:280; num_workers_list = [2 3 4 5 6 7 8]; jobid_list = 1; % locality_list = [20 25 30 35 40 45 50 55 60]; num_workers_list = [1 4 5 6 7 8 10]; % num_workers_list = [1 2 4 5 6 8 10 12];

%% previous
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
lineWidth = 1.5;
dt = 1/4; 
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
% n_data = size(trajectories_true, 1);
n_data = N;
times = repmat(0:dt*max_lyapunov:(n_steps-1)*dt*max_lyapunov, n_data, 1);
locations = repmat((1:n_data).', 1, n_steps);
threshold = 1.0;
% threshold = 0.5;
% threshold = 0.30;
% threshold = 0.10;
geo_mean = true;
T_th_max = round(1 / dt / max_lyapunov);
T_list = zeros(length(num_workers_list), length(locality_list));
T_list_geo = zeros(length(num_workers_list), length(locality_list)); %RMSEの相乗平均から計算した短期予測時間
RMSE_list = zeros(length(num_workers_list), length(locality_list)); %訓練誤差
Lambda_list = zeros(length(jobid_list), length(locality_list)); %
max_deltas = zeros(length(num_workers_list), length(locality_list)); %Δの最大値
lambda_error_list = zeros(length(num_workers_list), length(locality_list)); %一定時間後の誤差から計算したRMSEのリアプノフ指数
lambda_error_list2 = zeros(length(num_workers_list), length(locality_list)); %RMSEが閾値を超えた時間から計算したRMSEのリアプノフ指数
pred_error_list = zeros(length(num_workers_list), length(locality_list)); %予測時間
n_diverge_list = zeros(length(num_workers_list), length(locality_list)); %アトラクタを外れた回数、割合
for h = 1:length(num_workers_list)
    num_workers = num_workers_list(h);
    errors = zeros(length(locality_list), pred_length);
    train_errors = zeros(length(jobid_list), length(locality_list));
    % lambda_mean_list = zeros(1, length(jobid_list));
    lambda_delta_mean_list = zeros(1, length(locality_list));
    lambda_error_locality_list = zeros(length(locality_list), length(jobid_list));
    lambda_error_locality_list2 = zeros(length(locality_list), length(jobid_list));
    pred_error_locality_list = zeros(length(locality_list), length(jobid_list));
    PredTime_fix = zeros(length(jobid_list), length(locality_list));
    figure();
    for k = 1:length(locality_list)
        ln_error_list = zeros(1, length(jobid_list));
        T_error_list = zeros(1, length(jobid_list));
        ln_error_list2 = zeros(1, length(jobid_list));
        T_error_list2 = zeros(1, length(jobid_list));

        locality = locality_list(k);
        % T_trials = zeros(length(jobid_list), length(locality_list));
        T_trials = zeros(length(jobid_list), num_pred);

        t1_sum = 0;
        log_delta_sum = 0;
        for m = 1:length(jobid_list)
            jobid = jobid_list(m);
            Dr = Ntotal / num_workers;
            switch learn 
                case 'LSM_common'
                    % filename = ['\\nas08c093\data\otsuki\parallelized-reservoir-computing\KSParallelReservoir\KS\KS_result_LSM_common_uniform_train80000_node' num2str(Dr) '-L' num2str(L) '-radius0.6-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter1.mat'];
                    filename = ['\\nas08c093\data\otsuki\parallelized-reservoir-computing\KSParallelReservoir\' data_kind '\' data_kind '_Gradient_result_' learn '_' reservoir_kind '_reservoir_train' num2str(train_steps) '_node' num2str(Dr) '-L' num2str(L) '-N' num2str(N) '-radius0.6-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter1.mat'];
                
                    load(filename, 'error', 'RMSE_mean');
                case 'LSM_GD_short_prediction_time' 
                    if beta == 1e-4 % beta = 1e-4;
                        filename = ['\\nas08c093\data\otsuki\parallelized-reservoir-computing\KSParallelReservoir\' data_kind '\' data_kind '_Gradient_delta' num2str(del_locality) '_result_' learn '_' reservoir_kind '_reservoir_train' num2str(train_steps) '_node' num2str(Dr) '-L' num2str(L) '-N' num2str(N) '-radius0.6-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter1.mat'];
                    else % beta = 1e-2
                        filename = ['\\nas08c093\data\otsuki\parallelized-reservoir-computing\KSParallelReservoir\' data_kind '\' data_kind '_Gradient_delta' num2str(del_locality) '_result_' learn '_' reservoir_kind '_reservoir_' 'beta' num2str(beta) '_train' num2str(train_steps) '_node' num2str(Dr) '-L' num2str(L) '-N' num2str(N) '-radius0.6-locality' num2str(locality) '-numlabs' num2str(num_workers) '-jobid' num2str(jobid) '-index_iter1.mat'];
                    end
                    load(filename, 'TrainErrors', 'error_locality', 'del_locality', 'error', 'RMSE_mean', 'lambda_list', 'deltas','resparams', 'delta1_list');
                    
                    % % lambdas = (log(deltas(:,2:end)./deltas(:,1))./(2:size(deltas,2)));
                    % % Lambdas = mean(lambdas, 1);
                    % % Dth_list = 0.03;
                    % Dth_list = 0.1;
                    % % Dth_list = 0.01:0.04:0.09;
                    % % Dth_list = 0.01:0.02:0.09;
                    % Tth_list = 800;
                    % % Tth_list = 1000;
                    % % Tth_list = 10:200:1010;
                    % % Tth_list = 10:500:1010;
                    % % num_preds = 10;
                    % % num_preds = 20;
                    % % num_preds = 40;
                    % % fprintf('locality=%d\n', locality);
                    % lambda_list2 = zeros(length(Dth_list), length(Tth_list));
                    % for dd = 1:length(Dth_list)
                    %     Dth = Dth_list(dd);
                    %     for tt = 1:length(Tth_list)
                    %         Tth = Tth_list(tt);
                    %         t1list = zeros(num_pred, 1);
                    %         delta1_list = zeros(num_pred, 1);
                    %         % ln_mean = 0;
                    %         for kk = 1:num_pred
                    %             over_Dth_list = find(deltas(kk,:)>Dth);
                    %             if size(over_Dth_list, 2) == 0
                    %                 t1list(kk) = Tth;
                    %             elseif over_Dth_list(1) > Tth
                    %                 t1list(kk) = Tth;
                    %             else
                    %                 t1list(kk) = over_Dth_list(1);
                    %             end
                    %             delta1_list(kk) = deltas(kk, t1list(kk));
                    %             t1_sum = t1_sum + (t1list(kk)-1);
                    %             log_delta_sum = log_delta_sum + log10(delta1_list(kk) / deltas(kk, 1));
                    %             % th = 2e-3;
                    %             th = 1e-3;
                    %             if max(deltas(kk, :)) > th
                    %                 n_diverge_list(h, k) = n_diverge_list(h, k) + 1;
                    %             end
                    %         end
                    %         ln_list = log(delta1_list./deltas(1:num_pred, 1));
                    %         ln_mean = mean(ln_list);
                    %         t_mean = mean(t1list-1);
                    %         lambda = ln_mean / t_mean;
                    %         lambda_list2(dd, tt) = lambda;
                    %         % fprintf('Dth=%f, Tth=%f: lambda=%f\n', Dth, Tth, lambda);
                    %     end
                    % end
            end
            % train_errors(1, k) = train_errors(1, k) + RMSE_mean;
            % % % train_errors(m, k) =  RMSE_mean;
            train_errors(m, k) =  TrainErrors(2)-TrainErrors(1);
            
            % switch learn
            %     case 'LSM_GD_short_prediction_time'
            %         % if size(find(delta1_list<0.3), 1) == 0
            %         if size(find(delta1_list<0.2), 1) == 0
            %             Lambda_list(m, k) = 0;
            %         else 
            %             Lambda_list(m, k) = lambda_list(end, end);
            %         end
            %         % max_deltas(h, k) = max_deltas(h, k) + log10(max(max(deltas)));
            %         max_deltas(h, k) = max_deltas(h, k) + max(max(log10(deltas)));
            %         % max_deltas(h, k) = max_deltas(h, k) + mean(mean(log10(deltas)));
            % end
            % % pred_length = 2899; num_pred = 49;
            % % figure();
            % % num_pred = 49;
            T_list_forGradient = zeros(1, 2);
            for dlocality_index = 1:2
                error = error_locality(dlocality_index, :);
                pred_times_each_l = zeros(1, num_pred);
                for l = 1:num_pred % length(pred_marker_array)
                    if geo_mean
                        errors(k, :) = errors(k, :) + log10(error(1, (l-1)*pred_length+1:(l-1)*pred_length+pred_length));
                    else
                        errors(k, :) = errors(k, :) + error(1, (l-1)*pred_length+1:(l-1)*pred_length+pred_length);
                    end
                    error_pred = error(1, (l-1)*pred_length+1:(l-1)*pred_length+pred_length);
                    pred_time = find(error_pred>threshold);
                    pred_time = pred_time(1);
                    pred_times_each_l(l) = pred_time;
                    % % T_list(h, k) = pred_time(1);
                    T_trials(m, l) = pred_time;
                    % T_error_list2(m) = T_error_list2(m) + (pred_time-1)*dt; %%% dt kakeru???
                    % ln_error_list2(m) = ln_error_list2(m) + log(error_pred(pred_time)/error_pred(1));
    
                    % if pred_time > T_th_max
                    %     ln_error_list(m) = ln_error_list(m) + log(error_pred(T_th_max)/error_pred(1));
                    %     T_error_list(m) = T_error_list(m) + (T_th_max-1)*dt;
                    % else
                    %     ln_error_list(m) = ln_error_list(m) + log(error_pred(pred_time)/error_pred(1));
                    %     T_error_list(m) = T_error_list(m) + (pred_time-1)*dt;
                    % end
                    % pred_error_locality_list(k, m) = pred_error_locality_list(k, m) + error_pred(T_th_max);

                    % T_list_forGradient(dlocality_index) = T_list_forGradient(dlocality_index) + pred_time;
                end
                errors(k, :) = errors(k, :) / num_pred;
                errors(k, :) = 10.^errors(k, :);
                pred_time = find(errors(k,:)>threshold);
                pred_time = pred_time(1);
                % T_list_forGradient(dlocality_index) = pred_time;
                T_list_forGradient(dlocality_index) = median(pred_times_each_l);
            end
            PredTime_fix(m, k) = T_list_forGradient(2) - T_list_forGradient(1);
            % ln_error_list(m) = ln_error_list(m) / num_pred;
            % T_error_list(m) = T_error_list(m) / num_pred;
            % ln_error_list2(m) = ln_error_list2(m) / num_pred;
            % T_error_list2(m) = T_error_list2(m) / num_pred;
            % pred_error_locality_list(k, m) = pred_error_locality_list(k, m) / num_pred;
            RMSE_list(h, k) = RMSE_list(h, k) + RMSE_mean;
        end
        RMSE_list(h, k) = RMSE_list(h, k) / length(jobid_list);
        lambda_error_locality_list(k, :) = ln_error_list ./ T_error_list;
        lambda_error_locality_list2(k, :) = ln_error_list2 ./ T_error_list2;
        pred_error_list(h, k) = mean(pred_error_locality_list(k, :));
        
        max_deltas(h, k) = max_deltas(h, k) / length(jobid_list);
        switch learn
            case 'LSM_GD_short_prediction_time'
                log_delta_mean = log_delta_sum / (num_pred*length(jobid_list));
                t1_mean = t1_sum / (num_pred*length(jobid_list));
                lambda_delta_mean_list(k) = log_delta_mean / t1_mean;
        end
        errors(k, :) = errors(k, :) / (num_pred*length(jobid_list));
        if geo_mean
            errors(k, :) = 10.^errors(k, :);
        end
        % train_errors(1, k) = train_errors(1, k) / length(jobid_list);
        % figure();
        % errorbar(mean(T_trials, 2), std(T_trials, 0, 2));
        % xlabel('jobid'); ylabel('short prediction time');
        % title(['locality: ' num2str(locality)]);
        hold on;
        % plot(error);
        % plot(times(1,1:pred_length), errors(1, 1:pred_length), 'DisplayName', ['locality=' num2str(locality)]); 
        plot(times(1,1:pred_length), errors(k, 1:pred_length), 'LineWidth', lineWidth, 'DisplayName', ['l=' num2str(locality)]); 
        % plot(times(1,:), error(1, 1:n_steps), 'DisplayName', ['train steps=' num2str(train_steps)]); 
        hold off;
        
        % pred_time = find(errors(k, :)>threshold);
        % T_list(h, k) = pred_time(1);
        pred_time = median(median(T_trials));
        % pred_time = mean(mean(T_trials));
        T_list(h, k) = pred_time;
        out_times_geo = find(errors(k, :)>threshold);
        T_list_geo(h, k) = out_times_geo(1);
        % RMSE_list(h, :) = mean(train_errors, 1).';
    end
    lambda_error_list(h, :) = mean(lambda_error_locality_list, 2).';
    lambda_error_list2(h, :) = mean(lambda_error_locality_list2, 2).';
    % sgtitle(['L=' num2str(L) ', g=' num2str(request_pool_size) ', rho=' num2str(rho) ', D_r=' num2str(approx_reservoir_size)])
    sgtitle(['RMSE L=' num2str(L) ', num reservoirs: ' num2str(num_workers) ', Ntotal=' num2str(Ntotal)]);
    legend(); fontsize(16, 'points');
    % yticks(0:0.5:2.5); ylim([0 1.5]);
    xlabel('lyapunov time'); ylabel('Root Mean Squared Error');
    max_time = max(times(1,:));
    % xticks(0:floor(max_time/5/10)*10:max_time); 
    % axis tight; grid on; legend('Location', 'eastoutside');
    % xlim([0 6]);
    switch learn
        case 'LSM_GD_short_prediction_time'
            % figure(); 
            % hold on; plot(locality_list/N*L, mean(Lambda_list, 1));
            % scatter(locality_list/N*L, Lambda_list, 'black');
            % hold off; fontsize(16, 'points'); 
            % title(['g=' num2str(num_workers)]); xlabel('locality'); ylabel('\lambda'); grid on;
            % 
            % % figure(); errorbar(locality_list/N*L, mean(Lambda_list, 1), std(Lambda_list));
            % % xlabel('locality'); ylabel('\lambda');

            % figure(); scatter(locality_list/N*L, lambda_delta_mean_list); 
            % xlabel('locality'); ylabel('\lambda_{trans}'); title(['g=' num2str(num_workers)]);
            % fontsize(16, 'points'); grid on;
            % % figure(); scatter(locality_list/N*L, Lambda_list); xlabel('locality'); ylabel('\lambda_{trans}'); title(['g=' num2str(num_workers)]);
            % % fontsize(16, 'points');
    end
    % % show prediction time
    % figure(); plot(locality_list, T_list(h, :));
    % sgtitle(['L=' num2str(L) ', num reservoirs: ' num2str(num_workers) ', Ntotal=' num2str(Ntotal)]);
    % xlabel('locality'); ylabel('short-term prediction time');
    % legend(); fontsize(16, 'points'); axis tight; grid on;
end
%}

locality_list_one = [ones(length(locality_list), 1) locality_list.'];
b1 = locality_list_one \ PredTime_fix.';
b2 = locality_list_one \ train_errors.';

dL = 1/N*L;
figure(); scatter(L/N*locality_list, dt*max_lyapunov*PredTime_fix/dL);
% figure(); plot(L/N*locality_list, dt*max_lyapunov*PredTime_fix/dL);
% figure(); plot(L/N*locality_list, PredTime_fix);
% figure(); plot(mean(PredTime_fix)); 
xlabel('locality'); ylabel('gradient of pred time'); grid on;
% ylabel('\frac{\partial L}{\partial l}')
fontsize(16, 'points'); axis tight; grid on;
hold on;
plot(locality_list/N*L, dt*max_lyapunov*locality_list_one*b1/dL);
Rsq1 = 1 - sum((PredTime_fix-(locality_list_one*b1).').^2)/sum((PredTime_fix-mean(PredTime_fix)).^2);

figure(); scatter(L/N*locality_list, train_errors/dL);
% figure(); plot(L/N*locality_list, train_errors/dL);
% figure(); plot(mean(train_errors, 1));
xlabel('locality'); ylabel('gradient of train errors'); grid on;
fontsize(16, 'points'); axis tight; grid on;
hold on; 
plot(locality_list/N*L, locality_list_one*b2/dL);
Rsq2 = 1 - sum((train_errors-(locality_list_one*b2).').^2)/sum((train_errors-mean(train_errors)).^2);

figure(); 
for h = 1:length(num_workers_list)
    hold on;
    num_workers = num_workers_list(h);
    plot(locality_list/n_data*L, dt*max_lyapunov*T_list(h, :), 'DisplayName', ['g=' num2str(num_workers)]);
    hold off;
end
% sgtitle(['L=' num2str(L) ', Ntotal=' num2str(Ntotal)]);
% title(['g=' num2str(num_workers)]);
xlabel('locality')
ylabel('short-term prediction time (\Lambda_1t)');
fontsize(16, 'points'); axis tight; grid on;
% xlabel('locality (space)'); %xlabel('locality'); 
% ylabel('short-term prediction time');
% legend('Location', 'eastoutside'); 
% legend off;

Grad_TrainError_nonfix = RMSE_list(2:end)-RMSE_list(1:end-1);
Grad_PredictionTime_nonfix = T_list(2:end)-T_list(1:end-1);

locality_list_nonfix_one = [ones(length(locality_list(1:end-1)), 1) locality_list(1:end-1).'];
b1_nonfix = locality_list_nonfix_one \ Grad_PredictionTime_nonfix.';
b2_nonfix = locality_list_nonfix_one \ Grad_TrainError_nonfix.';
Rsq1_nonfix = 1 - sum((Grad_PredictionTime_nonfix-(locality_list_nonfix_one*b1_nonfix).').^2)/sum((Grad_PredictionTime_nonfix-mean(Grad_PredictionTime_nonfix)).^2);
Rsq2_nonfix = 1 - sum((Grad_TrainError_nonfix-(locality_list_nonfix_one*b2_nonfix).').^2)/sum((Grad_TrainError_nonfix-mean(Grad_TrainError_nonfix)).^2);

figure(); scatter(locality_list(1:end-1)/n_data*L, dt*max_lyapunov*Grad_TrainError_nonfix);
% figure(); plot(locality_list(1:end-1)/n_data*L, dt*max_lyapunov*Grad_TrainError_nonfix);
% figure(); errorbar(locality_list/n_data*L, mean(train_errors, 1), std(train_errors));
xlabel('locality'); ylabel('train error'); fontsize(16, 'points'); axis tight;
grid on;



% figure(); surf(repmat(num_workers_list.', 1, length(locality_list)), repmat(locality_list, length(num_workers_list), 1), T_list);
% xlabel('number of reservoirs'); ylabel('locality'); title(['L=' num2str(L)]);
% colorbar; view(0, 90); fontsize(16, 'points'); axis tight;

switch length(num_workers_list)
    case 1
        % figure(); plot(locality_list/n_data*L, train_errors);
        figure(); plot(locality_list/n_data*L, mean(train_errors, 1));
        % figure(); errorbar(locality_list/n_data*L, mean(train_errors, 1), std(train_errors));
        xlabel('locality_space'); ylabel('train error'); fontsize(16, 'points');
        grid on;

        figure(); 
        % num_workers = num_workers_list(h);
        plot(locality_list/n_data*L, T_list_geo(h, :), 'DisplayName', ['g=' num2str(num_workers)]);
        % sgtitle(['L=' num2str(L) ', Ntotal=' num2str(Ntotal)]);
        title(['g=' num2str(num_workers)]);
        xlabel('locality (space)'); %xlabel('locality'); 
        ylabel('short-term prediction time');
        legend('Location', 'eastoutside'); legend off;
        fontsize(16, 'points'); axis tight; grid on;

        figure(); plot(locality_list*L/N, RMSE_list);
        xlabel('locality'); ylabel('train error');
        grid on; fontsize(16, 'points'); axis tight;
        
        % figure(); plot(L/n_data*locality_list, pred_error_list);
        % xlabel('locality'); ylabel('pred error'); 
        % title(['pred error ' data_kind 'L=' num2str(L) ', fix g*Dr']);
        % fontsize(16, 'points'); axis tight;

        % figure(); plot(L/n_data*locality_list, lambda_error_list);
        % xlabel('locality'); ylabel('lambda error'); 
        % title(['lambda error ' data_kind 'L=' num2str(L) ', fix g*Dr']);
        % fontsize(16, 'points'); axis tight;

        % figure(); plot(L/n_data*locality_list, lambda_error_list2);
        % xlabel('locality'); ylabel('lambda error'); 
        % title(['lambda2 error ' data_kind 'L=' num2str(L) ', fix g*Dr']);
        % fontsize(16, 'points'); axis tight;
        % switch learn
        %     case 'LSM_GD_short_prediction_time'
        %         figure(); plot(locality_list/N*L, max_deltas); 
        %         xlabel('locality space'); ylabel('max deltas');
        % 
        %         figure(); plot(locality_list/N*L, n_diverge_list); 
        %         xlabel('locality space'); ylabel('n diverge'); grid on;
        % end
    otherwise
        figure(); surf(repmat(num_workers_list.', 1, length(locality_list)), L/n_data*repmat(locality_list, length(num_workers_list), 1), max_lyapunov*dt*T_list, 'FaceColor', 'interp');
        xlabel('number of reservoirs'); ylabel('locality'); title(['short-term prediction time' data_kind 'L=' num2str(L) ', fix g*Dr']);
        colorbar; view(0, 90); fontsize(16, 'points'); axis tight;
        
        % figure(); surf(repmat(num_workers_list.', 1, length(locality_list)), L/n_data*repmat(locality_list, length(num_workers_list), 1), max_lyapunov*dt*T_list_geo, 'FaceColor', 'interp');
        % xlabel('number of reservoirs'); ylabel('locality'); title(['short-term prediction time (geo)' data_kind 'L=' num2str(L) ', fix g*Dr']);
        % colorbar; view(0, 90); fontsize(16, 'points'); axis tight;
        
        figure(); surf(repmat(num_workers_list.', 1, length(locality_list)), L/n_data*repmat(locality_list, length(num_workers_list), 1), lambda_error_list, 'FaceColor', 'interp');
        xlabel('number of reservoirs'); ylabel('locality'); title(['\lambda error' data_kind 'L=' num2str(L) ', fix g*Dr']);
        colorbar; view(0, 90); fontsize(16, 'points'); axis tight;
        
        figure(); surf(repmat(num_workers_list.', 1, length(locality_list)), L/n_data*repmat(locality_list, length(num_workers_list), 1), lambda_error_list2, 'FaceColor', 'interp');
        xlabel('number of reservoirs'); ylabel('locality'); title(['\lambda2 error' data_kind 'L=' num2str(L) ', fix g*Dr']);
        colorbar; view(0, 90); fontsize(16, 'points'); axis tight;

        figure(); surf(repmat(num_workers_list.', 1, length(locality_list)), L/n_data*repmat(locality_list, length(num_workers_list), 1), pred_error_list, 'FaceColor', 'interp');
        xlabel('number of reservoirs'); ylabel('locality'); title(['pred error ' data_kind 'L=' num2str(L) ', fix g*Dr']);
        colorbar; view(0, 90); fontsize(16, 'points'); axis tight;

        figure(); surf(repmat(num_workers_list.', 1, length(locality_list)), L/n_data*repmat(locality_list, length(num_workers_list), 1), RMSE_list, 'FaceColor', 'interp');
        xlabel('number of reservoirs'); ylabel('locality'); title(['RMSE ' data_kind 'L=' num2str(L) ', fix g*Dr']);
        colorbar; view(0, 90); fontsize(16, 'points'); axis tight;
        switch learn
            case 'LSM_GD_short_prediction_time'
                figure(); surf(repmat(num_workers_list.', 1, length(locality_list)), L/n_data*repmat(locality_list, length(num_workers_list), 1), max_deltas,  'FaceColor', 'interp');
                xlabel('number of reservoirs'); ylabel('locality'); title(['delta ' data_kind 'L=' num2str(L) ', fix g*Dr']);
                colorbar; view(0, 90); fontsize(16, 'points'); axis tight;

                figure(); surf(repmat(num_workers_list.', 1, length(locality_list)), L/n_data*repmat(locality_list, length(num_workers_list), 1), n_diverge_list/(length(jobid_list)*num_pred),  'FaceColor', 'interp');
                xlabel('number of reservoirs'); ylabel('locality'); title(['n_diverge ' data_kind 'L=' num2str(L) ', fix g*Dr']);
                colorbar; view(0, 90); fontsize(16, 'points'); axis tight;
        end
end
% figure(); surf(repmat(num_workers_list.', 1, length(locality_list)), repmat(locality_list, length(num_workers_list), 1), T_list, 'FaceColor', 'interp', 'EdgeColor', 'interp');
% xlabel('number of reservoirs'); ylabel('locality'); title(['L=' num2str(L)]);
% colorbar; view(0, 90); fontsize(16, 'points'); axis tight;