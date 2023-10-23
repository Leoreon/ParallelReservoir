function prediction = predict(w,w_out,x,w_in,pl,chunk_size,frontWkrIdx, rearWkrIdx,N, locality)
prediction = zeros(chunk_size,pl);
if locality > chunk_size
    secFrontWkrIdx = mod(frontWkrIdx, numlabs) + 1;
    secRearWkrIdx = mod(rearWkrIdx-2, numlabs) + 1;
    display(frontWkrIdx);
    display(rearWkrIdx);
    display(secFrontWkrIdx);
    display(secRearWkrIdx);
    for i=1:pl
        x_augment = x;
        x_augment(2:2:N) = x_augment(2:2:N).^2;
        out = (w_out)*x_augment;
        labBarrier;
        rear_out = labSendReceive(frontWkrIdx ,rearWkrIdx, out(end-chunk_size+1:end));
        front_out = labSendReceive(rearWkrIdx, frontWkrIdx, out(1:chunk_size));
        
        second_rear_out = labSendReceive(secFrontWkrIdx, secRearWkrIdx, out(end-(locality-chunk_size)+1:end));
        second_front_out = labSendReceive(secRearWkrIdx, secFrontWkrIdx, out(1:(locality-chunk_size)));
        % fprintf('out: \n');
        % display(size(out));
        % display(size(front_out));
        % display(size(rear_out));
        % break;
        feedback = vertcat(second_rear_out, rear_out, out, front_out, second_front_out);
        x = tanh(w*x + w_in*feedback); 
        prediction(:,i) = out;
    end
else
    for i=1:pl
        x_augment = x;
        x_augment(2:2:N) = x_augment(2:2:N).^2;
        out = (w_out)*x_augment;
        labBarrier;
        rear_out = labSendReceive(frontWkrIdx ,rearWkrIdx, out(end-locality+1:end));
        front_out = labSendReceive(rearWkrIdx, frontWkrIdx, out(1:locality));
        % fprintf('out: \n');
        % display(size(out));
        % display(size(front_out));
        % display(size(rear_out));
        % break;
        feedback = vertcat(rear_out, out, front_out);
        x = tanh(w*x + w_in*feedback); 
        prediction(:,i) = out;
    end
end