function prediction = predict_among_reservoirs(w,w_out,x,w_in,w_among,pl,chunk_size,frontWkrIdx, rearWkrIdx,N, locality)
prediction = zeros(chunk_size,pl);
if locality > chunk_size
    secFrontWkrIdx = mod(frontWkrIdx, numlabs) + 1;
    secRearWkrIdx = mod(rearWkrIdx-2, numlabs) + 1;
    % display(frontWkrIdx);
    % display(rearWkrIdx);
    % display(secFrontWkrIdx);
    % display(secRearWkrIdx);
    for i=1:pl
        x_augment = x;
        x_augment(2:2:N) = x_augment(2:2:N).^2;
        out = (w_out)*x_augment;
        labBarrier;
        rear_x = labSendReceive(frontWkrIdx, rearWkrIdx, out(end-chunk_size+1:end));
        front_x = labSendReceive(rearWkrIdx, frontWkrIdx, out(1:chunk_size));
        labBarrier;
        second_rear_x = labSendReceive(secFrontWkrIdx, secRearWkrIdx, out(end-(locality-chunk_size)+1:end));
        second_front_x = labSendReceive(secRearWkrIdx, secFrontWkrIdx, out(1:(locality-chunk_size)));
        x_adjacent = vertcat(second_rear_x, rear_x, front_x, second_front_x);
        % labBarrier;
        % rear_out = labSendReceive(frontWkrIdx, rearWkrIdx, out(end-chunk_size+1:end));
        % front_out = labSendReceive(rearWkrIdx, frontWkrIdx, out(1:chunk_size));
        % labBarrier;
        % second_rear_out = labSendReceive(secFrontWkrIdx, secRearWkrIdx, out(end-(locality-chunk_size)+1:end));
        % second_front_out = labSendReceive(secRearWkrIdx, secFrontWkrIdx, out(1:(locality-chunk_size)));
        % fprintf('out: \n');
        % display(size(out));
        % display(size(front_out));
        % display(size(rear_out));
        % break;
        feedback = vertcat(second_rear_out, rear_out, out, front_out, second_front_out);
        x = tanh(w*x + w_in*feedback); 
        x = tanh(w*x + w_in*feedback + w_among*x_adjacent); 
        prediction(:,i) = out;
    end
else
    for i=1:pl
        x_augment = x;
        x_augment(2:2:N) = x_augment(2:2:N).^2;
        % display('ee');
        out = (w_out)*x_augment;
        % display('ff');
        % labBarrier;
        % rear_out = labSendReceive(frontWkrIdx ,rearWkrIdx, out(end-locality+1:end));
        % front_out = labSendReceive(rearWkrIdx, frontWkrIdx, out(1:locality));
        labBarrier;
        rear_x = labSendReceive(frontWkrIdx ,rearWkrIdx, x_augment(end-locality+1:end));
        front_x = labSendReceive(rearWkrIdx, frontWkrIdx, x_augment(1:locality));
        x_adjacent = vertcat(rear_x, front_x);
        % fprintf('out: \n');
        % display(size(out));
        % display(size(front_out));
        % display(size(rear_out));
        % break;
        % feedback = vertcat(rear_out, out, front_out);
        % display(size(w));
        % display(size(x));
        % display(size(w_in));
        % display(size(feedback));
        % x = tanh(w*x + w_in*feedback); 
        x = tanh(w*x + w_in*out + w_among*x_adjacent); 
        prediction(:,i) = out;
    end
end