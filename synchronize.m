function x = synchronize(W,x,w_in,data,prediction_marker,sync_length)
x = zeros(size(x));
% size(w_in)
% size(data(:, 1))
for i=1:sync_length
    x = tanh(W*x + w_in*data(:,prediction_marker + i));
end
