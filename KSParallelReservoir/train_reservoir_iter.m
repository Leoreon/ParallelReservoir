%% train reservoir using multiple data with different initial values
function [x, wout, A, win, RMSE] = train_reservoir(resparams, data, labindex, jobid, locality, chunk_size)

[num_inputs,~] = size(data);

A = generate_reservoir(resparams.N, resparams.radius, resparams.degree, labindex, jobid);
q = resparams.N/num_inputs;
win = zeros(resparams.N, num_inputs);
for i=1:num_inputs
    rng(i)
    ip = (-1 + 2*rand(q,1));
    win((i-1)*q+1:i*q,i) = ip;
end
win = sparse(win);

states = reservoir_layer_iter(A, win, data, resparams);

% states(2:2:resparams.N,:) = states(2:2:resparams.N,:).^2;
n_iter = size(states, 2) / resparams.train_length;
data_index = zeros(n_iter, resparams.train_length);
for kk = 1:n_iter
    start = (resparams.discard_length+resparams.train_length)*kk - resparams.train_length + 1;
    data_index(kk, :) = start:start+resparams.train_length-1;
end
data_index = reshape(data_index, 1, []);
% wout = fit(resparams, states, data(locality+1:locality+chunk_size,resparams.discard_length + 1:resparams.discard_length + n_iter*resparams.train_length));
wout = fit(resparams, states, data(locality+1:locality+chunk_size,data_index));

x = states(:,end);

error = wout*states - data(locality+1:locality+chunk_size,data_index);
% error = wout*states - data(locality+1:locality+chunk_size,resparams.discard_length + 1:resparams.discard_length + resparams.train_length);
error = error .^ 2;
RMSE = sqrt(mean(mean(error)));
