function forward_overlap = indexing_function_forward(chunk_end, locality, num_inputs, chunk_size, data_kind, l)
switch data_kind
    case 'LCD'
        interval = 1;
        forward_overlap = chunk_end+1:interval:chunk_end+interval*locality;
        forward_overlap(forward_overlap<=0|forward_overlap>num_inputs) = [];
        % forward_overlap = mod(forward_overlap-1, num_inputs) + 1;
    otherwise
        interval = 1;
        forward_overlap = chunk_end+1:interval:chunk_end+interval*locality;
        forward_overlap = mod(forward_overlap-1, num_inputs) + 1;
end

% if false % narrow connection
%     if chunk_end + locality <= num_inputs
%         forward_overlap = chunk_end+1:chunk_end+locality;
%     elseif chunk_end+locality>num_inputs && chunk_end == num_inputs
%         forward_overlap = 1:mod(chunk_end + locality, num_inputs);
%     elseif chunk_end+locality>num_inputs && chunk_end < num_inputs
%         forward_overlap = horzcat(chunk_end+1:num_inputs, 1:mod(chunk_end+locality, num_inputs));
%     else 
%         forward_overlap = -NaN;
%     end
% elseif true % broader connection
    % interval = floor((num_inputs-chunk_size)/2/locality/4);
    % if chunk_end + interval * locality <= num_inputs
    %     forward_overlap = chunk_end+1:interval:chunk_end+interval*locality;
    % elseif chunk_end+interval*locality > num_inputs && chunk_end == num_inputs
    %     forward_overlap = 1:interval:mod(chunk_end + interval*locality, num_inputs);
    % elseif chunk_end+interval*locality > num_inputs && chunk_end < num_inputs
    %     forward_overlap = horzcat(chunk_end+1:interval:num_inputs, 1:interval:mod(chunk_end+interval*locality, num_inputs));
    % else 
    %     forward_overlap = -NaN;
    % end
%     interval = 1;
%     forward_overlap = chunk_end+1:interval:chunk_end+interval*locality;
%     forward_overlap = mod(forward_overlap-1, num_inputs) + 1;
% end