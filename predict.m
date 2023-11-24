function prediction = predict(w,w_out,x,w_in,pl,chunk_size,frontWkrIdx, rearWkrIdx,N, locality, num_kind_data, test_in)
prediction = zeros(chunk_size,pl);
inputExist = ~isempty(test_in); % inputがないなら0、あるなら1

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
        if num_kind_data == 6
            for k = 1:size(out, 1)/num_kind_data
                nxyz = out((k-1)*num_kind_data+2:(k-1)*num_kind_data+4);
                norm_ = norm(nxyz);
                out((k-1)*num_kind_data+2:(k-1)*num_kind_data+4) = nxyz / 2 / norm_;
            end
        end
        labBarrier;
        rear_out = labSendReceive(frontWkrIdx, rearWkrIdx, out(end-chunk_size+1:end));
        front_out = labSendReceive(rearWkrIdx, frontWkrIdx, out(1:chunk_size));
        labBarrier;
        second_rear_out = labSendReceive(secFrontWkrIdx, secRearWkrIdx, out(end-(locality*num_kind_data-chunk_size)+1:end));
        second_front_out = labSendReceive(secRearWkrIdx, secFrontWkrIdx, out(1:(locality*num_kind_data-chunk_size)));
        % fprintf('out: \n');
        % display(size(out));
        % display(size(front_out));
        % display(size(rear_out));
        % break;
        feedback = vertcat(second_rear_out, rear_out, out, front_out, second_front_out, test_in(:, (i):(i)*inputExist));
        % display(size(feedback));
        x = tanh(w*x + w_in*feedback); 
        prediction(:,i) = out;
    end
else
    for i=1:pl
        x_augment = x;
        x_augment(2:2:N) = x_augment(2:2:N).^2;
        % display('ee');
        out = (w_out)*x_augment;
        % display(num_kind_data);
        if num_kind_data == 6
            for k = 1:size(out, 1)/num_kind_data
                nxyz = out((k-1)*num_kind_data+2:(k-1)*num_kind_data+4);
                norm_ = norm(nxyz);
                out((k-1)*num_kind_data+2:(k-1)*num_kind_data+4) = nxyz / 2 / norm_;
            end
        end
        % display(norm(out(2:4)));
        % display('ff');
        labBarrier;
        rear_out = labSendReceive(frontWkrIdx ,rearWkrIdx, out(end-locality*num_kind_data+1:end));
        front_out = labSendReceive(rearWkrIdx, frontWkrIdx, out(1:locality*num_kind_data));
        % fprintf('out: \n');
        % display(size(out));
        % display(size(front_out));
        % display(size(rear_out));
        % break;
        feedback = vertcat(rear_out, out, front_out, test_in(:, i:(i)*inputExist));
        % display(size(feedback));
        % display(size(rear_out));
        % display(size(out));
        % display(size(front_out));
        % feedback = vertcat(rear_out, out, front_out);
        % display(size(w));
        % display(size(x));
        % display(size(w_in));
        % display(size(feedback));
        x = tanh(w*x + w_in*feedback); 
        prediction(:,i) = out;
    end
end

function norm_ = norm(vec)
    norm_ = sqrt(sum(vec.*vec));
end
end
