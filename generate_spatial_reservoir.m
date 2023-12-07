function A = generate_spatial_reservoir(N, radius, degree, labindex, jobid, nodes_per_input, locality)
% display('start generate_spatial_reservoir');
rng(labindex+jobid)

% locality = 10;
% sparsity = degree/size;
sparsity = double(degree)/(2*double(locality)+double(nodes_per_input));
% display((2*double(locality)+double(nodes_per_input)));
% sparsity = double(degree)/((2*double(locality)+1)*double(nodes_per_input));
% display(degree);
% display((2*locality+1)*nodes_per_input);
% display(sparsity);
% display(locality);
% display(double(nodes_per_input));
% display(class(nodes_per_input));
% display(class(locality));
% display(class(sparsity));
A = zeros(N, N);
% display(class(A)); 
for k = 1:N/nodes_per_input
    row_index = nodes_per_input*(k-1)+1:nodes_per_input*k;
    % col_index = mod((nodes_per_input*(k-locality-1)+1:nodes_per_input*(k+locality))-1, size) + 1;
    col_index = mod(nodes_per_input*(k-1)-locality+1:nodes_per_input*k+locality, N) + 1;
    % part = sprand(double(nodes_per_input), double(nodes_per_input*(2*locality+1)), double(sparsity));
    part = sprand(double(nodes_per_input), double(nodes_per_input+2*locality), double(sparsity));
    % display(class(part));
    A(row_index, col_index) = part;
    % A(row_index, col_index) = ones(nodes_per_input, nodes_per_input*(2*locality+1));
end
% A = sprand(size, size, sparsity);
% display(size(col_index));

A = sparse(A);
e = max(abs(eigs(A)));

A = (A./e).*radius;
% A = sparse(A);
% display('end generate_spatial_reservoir');