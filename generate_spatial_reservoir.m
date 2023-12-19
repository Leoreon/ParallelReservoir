function A = generate_spatial_reservoir(N, radius, degree, labindex, jobid, nodes_per_input, width)
% display('start generate_spatial_reservoir');
% rng(labindex+jobid)

% % locality = 10;
% % sparsity = degree/N;
% sparsity = double(degree)/(2*double(width)+double(nodes_per_input));
% display((2*double(width)+double(nodes_per_input)));
% sparsity = double(degree)/((2*double(width)+1)*double(nodes_per_input));
% display(degree);
% display((2*width+1)*nodes_per_input);
% display(sparsity);
% display(width);
% display(double(nodes_per_input));
% display(class(nodes_per_input));
% display(class(width));
% display(class(sparsity));
A = zeros(N, N);
% % display(class(A)); 
% for k = 1:N/nodes_per_input
%     row_index = nodes_per_input*(k-1)+1:nodes_per_input*k;
%     % col_index = mod((nodes_per_input*(k-locality-1)+1:nodes_per_input*(k+locality))-1, size) + 1;
%     col_index = mod(nodes_per_input*(k-1)-width+1:nodes_per_input*k+width, N) + 1;
%     % part = sprand(double(nodes_per_input), double(nodes_per_input*(2*locality+1)), double(sparsity));
%     part = sprand(double(nodes_per_input), double(nodes_per_input+2*width), double(sparsity));
%     % display(class(part));
%     A(row_index, col_index) = part;
%     % A(row_index, col_index) = ones(nodes_per_input, nodes_per_input*(2*locality+1));
% end
index_all = zeros((2*width+1)*N, 1);
sparsity = degree / (2*width+1);
for k = 1:N
    index_row = k-width:k+width;
    index_row(index_row<=0) = index_row(index_row<=0) + N;
    index_row(index_row>N) = index_row(index_row>N) - N;
    index_all((k-1)*(2*width+1)+1:k*(2*width+1)) = (k-1)*N + index_row;
end
A(index_all) = sprand(size(index_all, 1), 1, double(sparsity));
% A(index_all) = ones(size(index_all, 1), 1);
% A = sprand(size, size, sparsity);
% display(size(col_index));

A = sparse(A);
e = max(abs(eigs(A)));

A = (A./e).*radius;
% A = sparse(A);
% display('end generate_spatial_reservoir');