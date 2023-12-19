function states = reservoir_layer_pred(x, A, win, train_data, resparams, test_steps, train_in, sync_length)
n_steps = size(train_data, 2)-sync_length;
states = zeros(resparams.N, n_steps);
% x = zeros(resparams.N,1);

if isempty(train_in)
    all_input = train_data;
else
    all_input = vertcat(train_data, train_in);
    % all_input = [train_data; train_in];
end

% x = synchronize(W,x,w_in,data,prediction_marker,sync_length,test_in)
x = synchronize(A, x, win, train_data, 1, sync_length, train_in);
% for i = 1:resparams.discard_length
%     % x = tanh(A*x + win*input(:,i));
%     % x = tanh(A*x + win*[input(:,i); train_in(:,i)]);
%     x = tanh(A*x + win*all_input(:,i));
% end

states(:,1) = x;
% for i = 1:resparams.train_length-1
for i = 1:n_steps-1
    % states(:,i+1) = tanh(A*states(:,i) + win*input(:,resparams.discard_length + i));
    % states(:,i+1) = tanh(A*states(:,i) + win*all_input(:,resparams.discard_length + i));
    states(:,i+1) = tanh(A*states(:,i) + win*all_input(:,sync_length+i));
    states(2:2:resparams.N,i) = states(2:2:resparams.N,i).^2;
end
states(2:2:resparams.N,end) = states(2:2:resparams.N,end).^2;

