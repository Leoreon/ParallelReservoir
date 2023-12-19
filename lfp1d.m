
% load('topographic1d_lz.mat');
% 
% % figure(); surf(lz); view(0, 90); shading interp; axis tight;
% lz_mean = movmean(lz, 50);
% lz_mean = lz_mean(1:50:end,:);
% figure(); surf(lz_mean); view(0, 90); shading interp; axis tight;
% 
% % lz_fourier
% F = fftshift(abs(fft2(lz)));
% S = fftshift(abs(fft2( lz(randperm(size(lz,1)), :) ) ));
% T = fftshift(abs(fft2(lz(:, randperm(size(lz,2))))));
% A = 10*log10((2*F) ./ (S+T));

load('topographic1d_lz2.mat');
lz2_fourier = fftshift(abs(fft2(lz2)));
figure(); surf(lz2_fourier); view(0, 90); shading interp; axis tight;