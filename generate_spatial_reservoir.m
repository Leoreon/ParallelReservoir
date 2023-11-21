function A = generate_spatial_reservoir(size, radius, degree, labindex, jobid, nodes_per_input)
tic
rng(labindex+jobid)

locality = 10;
% sparsity = degree/size;
sparsity = degree/((2*locality+1)*nodes_per_input);

A = zeros(size, size);
for k = 1:size/nodes_per_input
    row_index = nodes_per_input*(k-1)+1:nodes_per_input*k;
    col_index = mod((nodes_per_input*(k-locality-1)+1:nodes_per_input*(k+locality))-1, size) + 1;
    A(row_index, col_index) = sprand(nodes_per_input, nodes_per_input*(2*locality+1), sparsity);
    % A(row_index, col_index) = ones(nodes_per_input, nodes_per_input*(2*locality+1));
end
% A = sprand(size, size, sparsity);


% A = sparse(A);
e = max(abs(eigs(A)));

A = (A./e).*radius;
% A = sparse(A);
toc