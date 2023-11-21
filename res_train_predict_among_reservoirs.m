function [pred_collect, RMSE] = res_train_predict_among_reservoirs(in, test_in, resparams, jobid, locality, chunk_size, pred_marker_array, sync_length, nodes_per_input)


[x, w_out, w, w_in, w_among, RMSE] = train_reservoir_among_reservoirs(resparams, in, labindex, jobid, locality, chunk_size, nodes_per_input);
toc
fprintf('finished training\n');
frontWkrIdx = mod(labindex, numlabs) + 1; % one worker to the front
rearWkrIdx = mod(labindex - 2, numlabs) + 1; % one worker to the rear

num_preds = length(pred_marker_array);

pred_collect = zeros(chunk_size,num_preds*resparams.predict_length);

for pred_iter = 1:num_preds

    prediction_marker = pred_marker_array(pred_iter);
    
    x = synchronize(w,x,w_in,test_in,prediction_marker,sync_length);

    prediction = predict_among_reservoirs(w,w_out,x,w_in,w_among,resparams.predict_length,chunk_size, frontWkrIdx, rearWkrIdx,resparams.N, locality);

    pred_collect(:,(pred_iter-1)*resparams.predict_length+1:pred_iter*resparams.predict_length) = prediction;
    
end

