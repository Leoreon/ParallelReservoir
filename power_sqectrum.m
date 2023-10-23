
%% plot of power spectrum of KS
% L = 200; N = 512;
L = 400; N = 1024;
% L = 400; N = 2048;
% L = 800; N = 2048;
% load(['KS_science_L' num2str(L) '_N_' num2str(N) '_dps20000.mat']);
load(['KS_L' num2str(L) '_N_' num2str(N) '_dps20000.mat']);
% test_input_sequence = test_input_sequence.';

sum_p = zeros(4096, 1);
naverage = 500;
interval = 20;
for i =2000:interval:2000+interval*naverage-1
    % [p, f] = pspectrum(test_input_sequence(i,:), N/L*2*pi);
    [p, f] = pspectrum(test_input_sequence(i,:), N/L);
    % p = pspectrum(test_input_sequence(i,:), 128/50);
    % pp = 20*log10(p);

    % figure(); plot(f, p);
    % set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log');
    % xlabel('Spatial Frequency (Hz)'); ylabel('Power Spectrum(dB)');
    % fontsize(16, 'points');
    sum_p = sum_p + p;
end
sum_p = sum_p / naverage;
% sum_p = sum_p .^ (1/naverage);

figure();
scatter(2*pi*f, sum_p);
% plot(2*pi*f, sum_p);
set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log');
% xlabel('Spatial Frequency (Hz)'); 
% ylabel('Power Spectrum');
fontsize(16, 'points');
% title(['average over ' num2str(naverage)]);
title(['Powerspectrum ' 'L=' num2str(L)]);
xlim([10^-2 10^1]); ylim([10^-4 10^2]);
xticks(logspace(-2, 1, 4)); yticks(logspace(-4, 2, 7));
xlabel('q'); ylabel('g_q');
grid on;

[max_value, max_index] = max(sum_p)
max_freq = f(max_index);
fprintf('max frequency: %f\n', max_freq);
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
%}
