% function [x, wout, A, win, RMSE] = train_reservoir(resparams, data, labindex, jobid, locality, n_kind_data, chunk_size, nodes_per_input, train_in)
function [x, wout, A, win, RMSE] = train_reservoir(resparams, data, labindex, jobid, rear_locality_data, forward_locality_data, n_kind_data, chunk_size, nodes_per_input, train_in)

[num_inputs,~] = size([data; train_in]);
% [num_inputs,~] = size(data);

A = generate_reservoir(resparams.N, resparams.radius, resparams.degree, labindex, jobid);
% q = resparams.N/num_inputs;

win = zeros(resparams.N, num_inputs);
n_additional = rem(resparams.N, num_inputs);
nodes_per_in = floor(double(resparams.N) / double(num_inputs));
% display('eee');
add_id = randsample(num_inputs, n_additional);
% display('eeeq');
nodes_list = double(nodes_per_in) * ones(1, num_inputs);
% display('fff');
nodes_list(add_id) = nodes_list(add_id) + 1;
beg = 1;
for i=1:num_inputs
    rng(i)
    % ip = (-1 + 2*rand(q,1));
    % win((i-1)*q+1:i*q,i) = ip;
    ip = (-1 + 2*rand(nodes_list(i), 1));
    fin = beg + nodes_list(i)-1;
    win(beg:fin,i) = ip;
    beg = fin + 1;
end
% display(size(win));
% A = generate_reservoir(resparams.N, resparams.radius, resparams.degree, labindex, jobid);
% % A = generate_spatial_reservoir(resparams.N, resparams.radius, resparams.degree, labindex, jobid, nodes_per_input);
% q = resparams.N/num_inputs;
% display(num_inputs);
% win = zeros(resparams.N, num_inputs);
% for i=1:num_inputs
%     rng(i)
%     ip = (-1 + 2*rand(q,1));
%     if true % 'uniform'
%         % if i <= locality || i > num_inputs - locality
%         %     ip = ip / 4;
%         % end
%         ip = ip;
%     elseif false % linear
%         if i <= locality
%             ip = ip * i / locality;
%         elseif i > num_inputs - locality
%             ip = ip * (num_inputs-i+1) / locality;
%         end
%     end
%     win((i-1)*q+1:i*q,i) = ip;
% end
win = sparse(win);

states = reservoir_layer(A, win, data, resparams, train_in);

% states(2:2:resparams.N,:) = states(2:2:resparams.N,:).^2;
% display(size(states));
% display(size(data));
% wout = fit(resparams, states, data(n_kind_data*locality+1:n_kind_data*locality+chunk_size,resparams.discard_length + 1:resparams.discard_length + resparams.train_length));
wout = fit(resparams, states, data(rear_locality_data+1:rear_locality_data+chunk_size,resparams.discard_length + 1:resparams.discard_length + resparams.train_length));
% display(size(wout));
x = states(:,end);

error = wout*states - data(rear_locality_data+1:rear_locality_data+chunk_size,resparams.discard_length + 1:resparams.discard_length + resparams.train_length);
error = error .^ 2;
RMSE = sqrt(mean(mean(error)));
