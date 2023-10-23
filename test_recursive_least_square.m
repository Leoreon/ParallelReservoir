jobid = 1;
data_dir = './';
% data_kind = 'KS';
data_kind = 'CGL';
switch data_kind
case 'CGL'
    L = 8; N = 64; train_steps = 100000; test_steps = 20000;
    % L = 8; N = 32; train_steps = 80000; test_steps = 20000;
    m = matfile([data_dir 'CGL_L', num2str(L) '_N_', num2str(N) '_dps', num2str(train_steps) '.mat']); % CGL
    tf = matfile([data_dir 'CGL_L', num2str(L) '_N_' num2str(N) '_dps' num2str(test_steps) '.mat']); % CGL
case 'KS'
    m = matfile([data_dir 'train_input_sequence.mat']); % KS
    tf = matfile([data_dir 'test_input_sequence.mat']); % KS
end

rho = 0.6; 

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

locality = 6; % there are restrictions on the allowed range of this parameter. check documentation
% locality = locality_array; % there are restrictions on the allowed range of this parameter. check documentation

rear_overlap = indexing_function_rear(chunk_begin, locality, num_inputs, chunk_size);  %spatial overlap on the one side

forward_overlap = indexing_function_forward(chunk_end, locality, num_inputs, chunk_size);  %spatial overlap on the other side

overlap_size = length(rear_overlap) + length(forward_overlap); 

% approx_reservoir_size = 1000;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
approx_reservoir_size = 5000;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
% approx_reservoir_size = 7000;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)
% approx_reservoir_size = 9000;  % number of nodes in an individual reservoir network (approximate upto the next whole number divisible by number of inputs)

avg_degree = 3; %average connection degree

resparams.sparsity = avg_degree/approx_reservoir_size;

resparams.degree = avg_degree;

nodes_per_input = round(approx_reservoir_size/(chunk_size+overlap_size));

resparams.N = nodes_per_input*(chunk_size+overlap_size); % exact number of nodes in the network

resparams.train_length = 79000;  %number of time steps used for training
% resparams.train_length = 79000;  %number of time steps used for training
% resparams.train_length = 39000;  %number of time steps used for training

resparams.discard_length = 1000;  %number of time steps used to discard transient (generously chosen)

resparams.predict_length = 2999;  %number of steps to be predicted

sync_length = 5; % use a short time series to synchronize to the data at the prediction_marker
% sync_length = 32; % use a short time series to synchronize to the data at the prediction_marker
% sync_length = 100; % use a short time series to synchronize to the data at the prediction_marker

% resparams.radius = 0.6; % spectral radius of the reservoir
resparams.radius = rho; % spectral radius of the reservoir

resparams.beta = 0.0001; % ridge regression regularization parameter

u = zeros(len, chunk_size + overlap_size); % this will be populated by the input data to the reservoir

u(:,1:locality) = train_uu(1:end, rear_overlap);

u(:,locality+1:locality+chunk_size) = train_uu(1:end, chunk_begin:chunk_end);

u(:,locality+chunk_size+1:2*locality+chunk_size) = train_uu(1:end,forward_overlap);

u = sigma*u;

test_uu = tf.test_input_sequence;
% clear tf;
tf = [];

test_u = zeros(20000, chunk_size + overlap_size); % this will be populated by the input data to the reservoir

test_u(:,1:locality) = test_uu(1:end, rear_overlap);

test_u(:,locality+1:locality+chunk_size) = test_uu(1:end, chunk_begin:chunk_end);

test_u(:,locality+chunk_size+1:2*locality+chunk_size) = test_uu(1:end,forward_overlap);

test_u = sigma*test_u;
% fprintf('start res train predict');

data = u(1:600,:).';

[num_inputs,~] = size(data);

A = generate_reservoir(resparams.N, resparams.radius, resparams.degree, labindex, jobid);
q = resparams.N/num_inputs;
win = zeros(resparams.N, num_inputs);
for i=1:num_inputs
    rng(i)
    ip = (-1 + 2*rand(q,1));
    win((i-1)*q+1:i*q,i) = ip;
end

wout = zeros(chunk_size, resparams.N);
[x, wout] = recursive_least_square(resparams, data, win, A, wout, locality, chunk_size, sync_length);
% states = reservoir_layer(A, win, data, resparams);

% states(2:2:resparams.N,:) = states(2:2:resparams.N,:).^2;
% 
% wout = fit(resparams, states, data(locality+1:locality+chunk_size,resparams.discard_length + 1:resparams.discard_length + resparams.train_length));
% 
% x = states(:,end);
% 
% error = wout*states - data(locality+1:locality+chunk_size,resparams.discard_length + 1:resparams.discard_length + resparams.train_length);
% error = error .^ 2;
% RMSE = sqrt(mean(mean(error)));


% [pred_collect, RMSE] = res_train_predict(transpose(u), transpose(test_u), resparams, jobid, locality, chunk_size, pred_marker_array, sync_length);
% % fprintf('pred_collect');
% collated_prediction = gcat(pred_collect,1,1);