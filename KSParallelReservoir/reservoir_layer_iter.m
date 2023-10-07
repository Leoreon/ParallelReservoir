%% update reservoir using multiple data with different initial values
function states = reservoir_layer_iter(A, win, input, resparams)
whole_steps = size(input, 2);
n_iter = whole_steps / (resparams.train_length+resparams.discard_length);

states = zeros(resparams.N, n_iter*resparams.train_length);
x = zeros(resparams.N,1);
% size(states)
% n_iter
% resparams.train_length
for k = 1:n_iter
    
    for i = 1:resparams.discard_length
        x = tanh(A*x + win*input(:,(k-1)*resparams.train_length+i));
    end
    
    states(:,(k-1)*resparams.train_length+1) = x;
    
    for i = 1:resparams.train_length-1
        states(:,(k-1)*resparams.train_length+i+1) = tanh(A*states(:,(k-1)*resparams.train_length+i) + win*input(:,(k-1)*resparams.train_length+resparams.discard_length + i));
        states(2:2:resparams.N,(k-1)*resparams.train_length+i) = states(2:2:resparams.N,(k-1)*resparams.train_length+i).^2;
    end
    states(2:2:resparams.N,end) = states(2:2:resparams.N,end).^2;
end
% size(states)
