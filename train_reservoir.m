function [x, wout, A, win, RMSE] = train_reservoir(resparams, data, labindex, jobid, locality, chunk_size)

[num_inputs,~] = size(data);

A = generate_reservoir(resparams.N, resparams.radius, resparams.degree, labindex, jobid);
q = resparams.N/num_inputs;
win = zeros(resparams.N, num_inputs);
for i=1:num_inputs
    rng(i)
    ip = (-1 + 2*rand(q,1));
    if true % 'uniform'
        % if i <= locality || i > num_inputs - locality
        %     ip = ip / 4;
        % end
        ip = ip;
    elseif true % linear
        if i <= locality
            ip = ip * i / locality;
        elseif i > num_inputs - locality
            ip = ip * (num_inputs-i+1) / locality;
        end
    end
    win((i-1)*q+1:i*q,i) = ip;
end
win = sparse(win);

states = reservoir_layer(A, win, data, resparams);

% states(2:2:resparams.N,:) = states(2:2:resparams.N,:).^2;

wout = fit(resparams, states, data(locality+1:locality+chunk_size,resparams.discard_length + 1:resparams.discard_length + resparams.train_length));

x = states(:,end);

error = wout*states - data(locality+1:locality+chunk_size,resparams.discard_length + 1:resparams.discard_length + resparams.train_length);
error = error .^ 2;
RMSE = sqrt(mean(mean(error)));
