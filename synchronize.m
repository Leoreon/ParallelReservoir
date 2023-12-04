function x = synchronize(W,x,w_in,data,prediction_marker,sync_length,test_in)
x = zeros(size(x));
% size(w_in)
% size(data(:, 1))
if ~isempty(test_in)
    all_data = vertcat(data, test_in);
    % all_data = [data; test_in];
else
    all_data = data;
end
% display(size(all_data));
% display(size(W));
% display(size(x));
for i=1:sync_length
    % x = tanh(W*x + w_in*data(:,prediction_marker + i));
    % x = tanh(W*x + w_in*[data(:,prediction_marker + i); test_in(prediction_marker + i)]);
    x = tanh(W*x + w_in*all_data(:,prediction_marker + i));
end
