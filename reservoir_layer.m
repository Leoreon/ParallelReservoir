function states = reservoir_layer(A, win, train_data, resparams, train_in)

states = zeros(resparams.N, resparams.train_length);
x = zeros(resparams.N,1);

if isempty(train_in)
    all_input = train_data;
else
    all_input = vertcat(train_data, train_in);
    % all_input = [train_data; train_in];
end

for i = 1:resparams.discard_length
    % x = tanh(A*x + win*input(:,i));
    % x = tanh(A*x + win*[input(:,i); train_in(:,i)]);
    x = tanh(A*x + win*all_input(:,i));
end

states(:,1) = x;
for i = 1:resparams.train_length-1
    % states(:,i+1) = tanh(A*states(:,i) + win*input(:,resparams.discard_length + i));
    states(:,i+1) = tanh(A*states(:,i) + win*all_input(:,resparams.discard_length + i));
    states(2:2:resparams.N,i) = states(2:2:resparams.N,i).^2;
end
states(2:2:resparams.N,end) = states(2:2:resparams.N,end).^2;

