function [p, f] = power_u(u_x_t, L, N)
% u: space*time
% u_q_t: freq*time
% u = test_input_sequence.';
num_freq = 1000;
% num_steps = size(u);
% f = logspace(-2, 1, num_freq);
f = linspace(1e-2, 1e1, num_freq);
dx = L/N;
xs = 0:dx:L-dx;
wave = exp(-j*f.' .* xs);
u_q_t = 1/sqrt(L) *dx .* wave * u_x_t;
p = mean(abs(u_q_t).^2, 2);
end
