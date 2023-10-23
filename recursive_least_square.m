function [x, w_out] = recursive_least_square(resparams, data, w_in, A, w_out, locality, chunk_size, sync_length)
x = zeros(resparams.N, 1);
x = synchronize(A,x,w_in,data(:, 1:sync_length+1),1,sync_length);
% Rn = 0.7;
Rn = 1.0;
Pn = Rn * diag(ones(resparams.N, 1));
batch_size = 1;

targets = data(locality+1, :);
outputs = zeros(1, size(data, 2));
out = zeros(chunk_size, batch_size);
figure();
fprintf('start loop\n');
size(data)
for i = sync_length:size(data, 2)-1
    x_augment = x;
    x_augment(2:2:resparams.N) = x_augment(2:2:resparams.N).^2;
    out(:, mod(i-1, batch_size)+1) = (w_out)*x_augment;
    % labBarrier;
    % rear_out = labSendReceive(frontWkrIdx ,rearWkrIdx, out(end-locality+1:end));
    % front_out = labSendReceive(rearWkrIdx, frocvb ntWkrIdx, out(1:locality));
    % feedback = vertcat(rear_out, out, front_out);
    % x = tanh(A*x + w_in*feedback); 
    x = tanh(A*x + w_in*data(:,i)); 
    % prediction(:,i) = out;
    if mod(i, batch_size) == 0
        [w_out, Pn, Rn] = rls_w(w_out, Pn, Rn, data(locality+1:end-locality, i-batch_size+2:i+1), x_augment, out);
        % [w_out, Pn, Rn] = rls_w(w_out, Pn, Rn, data(locality+1:end-locality, i+1), x_augment);
    end
    outputs(i) = out(1);
    % if mod(i, 30) == 0
    %     plot(outputs);
    %     hold on
    %     plot(targets);
    %     hold off
    %     fprintf('plot\n');
    % end
    if mod(i, size(data, 2)/10)==0
        display(['i: ' num2str(i)]);
        fprintf('i/n_steps: %d / %d', i, size(data, 2));
    end
end
% plot(outputs);
% hold on
% plot(targets);
% hold off
% fprintf('plot\n');
