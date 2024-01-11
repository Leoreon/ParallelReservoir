
%% plot of power spectrum of KS
% L = 22; N = 64;
% L = 50; N = 128;
% L = 100; N = 256;
% L = 200; N = 512;
% L = 400; N = 1024;
% L = 400; N = 2048;
% L = 800; N = 2048;
% data_kind = 'KS';
data_kind = 'LFP';
L_list = 5.66e-3;
N_list = 120;
dps = 1000;
% L_list = 1e-3;
% N_list = 100;

% L_list = [22 26 30 34 38 44 50 100 200 400 800];
% N_list = [64 128 128 128 128 128 128 256 512 1024 2048];
% L_list = [22 50 100 200 400 800];
% N_list = [64 128 256 512 1024 2048];
% L_list = [22 44 66 88 200 400];
% N_list = [840 840 840 840 512 1024];
% L_list = [22 26 30 34 38 44 50];
% N_list = [64 128 128 128 128 128 128];
% L_list = [22];
% N_list = [2048];

% data_kind = 'CGL';
% L_list = [50 100 200];
% N_list = [32 64 128];
% c1 = -4; c2 = 1;
% c1 = -2; c2 = 2;
% c1 = -1; c2 = 2;
c1 = 0; c2 = -3;

% data_kind = 'LCD';
% L_list = 5e-6; N_list = 100;
figure();
powers = zeros(1000, length(L_list));
for k = 1:length(L_list)
    L = L_list(k); N = N_list(k);
    switch data_kind 
        case 'KS'
            % load(['KS_science_L' num2str(L) '_N_' num2str(N) '_dps20000.mat']);
            load(['KS_L' num2str(L) '_N_' num2str(N) '_dps' num2str(dps) '.mat']);
        case 'CGL'
            load([data_kind '_L' num2str(L) '_N_' num2str(N) '_dps20000' 'c1_' num2str(c1) 'c2_' num2str(c2) '.mat']);
        case 'LCD'
            test_dir = 'LCD_data/data/test';
            test_filename = [test_dir 'LCD_data.mat'];
            load(test_filename); L = 5e-6; N = 100;
        case 'LFP'
            load(['LFP_L' num2str(L) '_N' num2str(N) '_dps'  '.mat']);
            % load('Neurons_L0.001_N100_dps18000.mat');
    end
    % test_input_sequence = test_input_sequence.';
    
    % sum_p = zeros(4096, 1);
    naverage = 200;
    interval = 5;
    % interval = 20;
    [sum_p, f] = power_u(test_input_sequence(:, 1:3:end).', L, N);
    % for i =2000:interval:2000+interval*naverage-1
    %     % [p, f] = pspectrum(test_input_sequence(i,:), N/L*2*pi);
    %     [p, f] = pspectrum(test_input_sequence(i,:), N/L);
    %     % [p, f] = power(test_input_sequence(i, :));
    %     % p = pspectrum(test_input_sequence(i,:), 128/50);
    %     % pp = 20*log10(p);
    % 
    %     % figure(); plot(f, p);
    %     % set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log');
    %     % xlabel('Spatial Frequency (Hz)'); ylabel('Power Spectrum(dB)');
    %     % fontsize(16, 'points');
    %     sum_p = sum_p + p;
    % end
    % sum_p = sum_p / naverage;
    % % sum_p = sum_p .^ (1/naverage);
    
    powers(:, k) = sum_p;
    hold on;
    % scatter(2*pi*f, sum_p);
    plot(f, sum_p, 'DisplayName', ['L=' num2str(L)], 'LineWidth', 1.5);
    % plot(2*pi*f, sum_p, 'DisplayName', ['L=' num2str(L)]);
    hold off;
    set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log');
    xlabel('Spatial Frequency (Hz)'); 
    ylabel('Power');
    fontsize(16, 'points');
    % title(['average over ' num2str(naverage)]);
    switch data_kind
        case 'KS'
            title(['Powerspectrum of' ' KS']);
        case 'CGL'
            title(['Powerspectrum of CGL c_1=' num2str(c1) ', c_2=' num2str(c2)]);
    end
    % title(['Powerspectrum ' 'L=' num2str(L)]);
    % xlim([10^-2 10^0]); ylim([10^-4 10^2]);
    % xlim([10^-2 10^1]); ylim([10^-4 10^2]);
    % xticks(logspace(-2, 1, 4)); yticks(logspace(-4, 2, 7));
    % % xlabel('q'); ylabel('g_q');
    legend();
    grid on;
    
    [max_value, max_index] = max(sum_p);
    max_freq = f(max_index);
    fprintf('max frequency for L=%d: %f which is %f\n', L, max_freq/(2*pi), 2*pi/max_freq);
end

% corr = corrcoef(test_input_sequence(:, 1:end));
% figure(); plot(corr(1,:));
% figure(); plot(corr(end/2,:));
% axis tight
% grid;

% figure(); % plot(sum_p/30); 
% % pspectrum(test_input_sequence(3600,:), 128/50);
% [p, f] = pspectrum(test_input_sequence(3600,:), N/L);
% plot(f, p);
% set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log');
% xlabel('Spatial Frequency (Hz)'); ylabel('Power Spectrum(dB)');
% % xticks(0:50:200); yticks(-150:50:0);
% fontsize(16, 'points');
% % xticks(0:0.1:4); yticks(-140:20:0);
% % xlim([0 0.55]); ylim([-50 10]);

%{
%% color plot of CGL
load('CGL_L50_N_32_dps20000.mat');
test_input_sequence = test_input_sequence(1000:end,:).';
dt = 0.07; max_lyapunov = 1; data_kind = 'CGL'; L = 50; 
n_steps = 1000;
n_data = size(test_input_sequence, 1)/2;
times = repmat(0:dt*max_lyapunov:(n_steps-1)*dt*max_lyapunov, n_data, 1);
locations = repmat((1:n_data).', 1, n_steps);
max_value = max(max(test_input_sequence)); min_value = min(min(test_input_sequence));
figure(); 
subplot(2, 1, 1); surf(times, locations, test_input_sequence(1:2:end-1,1:n_steps)); 
view(0, 90); shading interp, axis tight; xlabel('lyapunov time'); ylabel('x');
title('Real part'); 
colorbar; clim([min_value max_value]); 
xlim([0 20]); xticks(0:10:20); yticks(0:n_data/2:n_data); yticklabels({0, L/2, L});
fontsize(16, 'points'); 

subplot(2, 1, 2); surf(times, locations, test_input_sequence(2:2:end,1:n_steps)); 
view(0, 90); shading interp, axis tight; xlabel('lyapunov time'); ylabel('x'); 
title('Imagenary part'); 
colorbar; clim([min_value max_value]); 
xlim([0 20]); xticks(0:10:20); yticks(0:n_data/2:n_data); yticklabels({0, L/2, L});
fontsize(16, 'points'); 
sgtitle('Complex Ginzburg-Landau equation'); 
% sgtitle([data_kind ' L:' num2str(L)]);

% figure(); surf(test_input_sequence.');
% xlim([0:]);

%% snapshot
L=88;
load(['KS_L' num2str(L) '_N_840_dps20000.mat']);
figure(); interval = L/840; space=0:interval:L-interval;
plot(space, test_input_sequence(1,:));
axis tight;
xlim([0 L]); ylim([-2.6 2.6]);
fontsize(16, 'points');
xlabel('x'); ylabel('y'); xticks(0:L/7:L);
grid on;
title(['L=' num2str(L)]);
%}

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
