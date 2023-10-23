function w_out = fit(params, states, data)
% size(data)
% size(states)
% params.N
beta = params.beta;

idenmat = beta*speye(params.N);

w_out = data*transpose(states)*pinv(states*transpose(states)+idenmat);
