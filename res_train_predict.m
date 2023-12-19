% function [pred_collect, RMSE] = res_train_predict(in, test_data, resparams, jobid, locality, n_kind_data, chunk_size, num_reservoirs_per_worker, pred_marker_array, sync_length, nodes_per_input, train_in, test_in)
function [pred_collect, RMSE] = res_train_predict(in, test_data, resparams, jobid, rear_locality_data, forward_locality_data, n_kind_data, chunk_size, num_reservoirs_per_worker, pred_marker_array, sync_length, nodes_per_input, train_in, test_in)
inputExist = ~isempty(test_in);
% display(inputExist);
% display(rear_locality_data);
% display(forward_locality_data);

% [x, w_out, w, w_in, RMSE] = train_reservoir(resparams, in, labindex, jobid, locality, n_kind_data, chunk_size, nodes_per_input, train_in);
[x, w_out, w, w_in, RMSE] = train_reservoir(resparams, in, labindex, jobid, rear_locality_data, forward_locality_data, n_kind_data, chunk_size, nodes_per_input, train_in);
% toc
fprintf('finished training\n');
frontWkrIdx = mod(labindex, numlabs) + 1; % one worker to the front
rearWkrIdx = mod(labindex - 2, numlabs) + 1; % one worker to the rear

num_preds = length(pred_marker_array);

pred_collect = zeros(chunk_size,num_preds*resparams.predict_length);

for pred_iter = 1:num_preds
    
    prediction_marker = pred_marker_array(pred_iter);
    
    x = synchronize(w,x,w_in,test_data,prediction_marker,sync_length, test_in);
    
    % prediction = predict(w,w_out,x,w_in,resparams.predict_length,chunk_size, num_reservoirs_per_worker, frontWkrIdx, rearWkrIdx,resparams.N, locality, n_kind_data,  test_in(:, prediction_marker+sync_length+1:inputExist*(prediction_marker+sync_length+resparams.predict_length)));
    prediction = predict(w,w_out,x,w_in,resparams.predict_length,chunk_size, num_reservoirs_per_worker, frontWkrIdx, rearWkrIdx,resparams.N, rear_locality_data, forward_locality_data, n_kind_data,  test_in(:, prediction_marker+sync_length+1:inputExist*(prediction_marker+sync_length+resparams.predict_length)));
    
    pred_collect(:,(pred_iter-1)*resparams.predict_length+1:pred_iter*resparams.predict_length) = prediction;
    
end

