function  rear_overlap = indexing_function_rear(chunk_begin, locality, num_inputs, chunk_size)
if false % narrow connection
    if chunk_begin - locality > 0
        rear_overlap = chunk_begin-locality:chunk_begin-1;
    elseif chunk_begin-locality <= 0 && chunk_begin > 1
        i1 = mod(chunk_begin - locality, num_inputs);
        rear_overlap = horzcat(i1:num_inputs, 1:chunk_begin-1);
    elseif chunk_begin-locality <= 0 && chunk_begin == 1
        i1 = mod(chunk_begin - locality, num_inputs);
        rear_overlap = i1:num_inputs;
    else 
        rear_overlap = -NaN; %throw error, hopefully :P
    end
elseif true % broader connection
    % interval = floor((num_inputs-chunk_size)/2/locality/4);
    % if chunk_begin - interval * locality > 0
    %     rear_overlap = chunk_begin-interval*locality:interval:chunk_begin-1;
    % elseif chunk_begin-interval*locality <= 0 && chunk_begin > 1
    %     i1 = mod(chunk_begin - interval*locality, num_inputs);
    %     rear_overlap = horzcat(i1:interval:num_inputs, 1:interval:chunk_begin-1);
    % elseif chunk_begin-interval*locality <= 0 && chunk_begin == 1
    %     i1 = mod(chunk_begin - interval*locality, num_inputs);
    %     rear_overlap = i1:interval:num_inputs;
    % else 
    %     rear_overlap = -NaN; %throw error, hopefully :P
    % end
    interval = 1;
    rear_overlap = chunk_begin-interval*locality:interval:chunk_begin-1;
    rear_overlap = mod(rear_overlap-1, num_inputs) + 1;
end
    