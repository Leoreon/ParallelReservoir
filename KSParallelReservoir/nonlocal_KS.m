clear;

%Generate data for the Kuramoto-Sivashinksy equation.
% Reference: Kassam, Aly-Khan, and Lloyd N. Trefethen.
% "Fourth-order time-stepping for stiff PDEs." 
% SIAM Journal on Scientific Computing 26.4 (2005): 1214-1233.


% N = 256;
N = 128;
% N = 64;
% d = 100;
% d = 88;
% d = 66;
d = 50;
% d = 44;
% d = 22;
x = d*(-N/2+1:N/2)'/N;
u = 0.6*(-1+2*rand(size(x)));
v = fft(u);
% Precompute various ETDRK4 scalar quantities:
h = 1/4; % time step
k = [0:N/2-1 0 -N/2+1:-1]'*(2*pi/d); % wave numbers
L = k.^2 - k.^4; % Fourier multipliers
E = exp(h*L); E2 = exp(h*L/2);
M = 16; % no. of points for complex means
r = exp(1i*pi*((1:M)-.5)/M); % roots of unity

LR = h*L(:,ones(M,1)) + r(ones(N,1),:);

%% parameters for Nonlocal term
% KK = 0;
KK=0.5;
% KK = 1;
XX    = (d/N)*(-N/2:N/2-1)'; 
% [X,Y] = meshgrid(XX);
X = XX;
% rr=X.*X+Y.*Y;
rr = X.*X;
K = besselk(0,sqrt(rr)); %K = besselk(0,sqrt(rr));
% K(N/2+1,N/2+1)=0;
K(N/2+1) = 0;
% sumK=sum(sum(K));
sumK = sum(K);
% K = circshift(K,[N/2 N/2]);
K = circshift(K, N/2);
K=K/sumK;
K_hat=fft2(K);

%%
Q = h*real(mean( (exp(LR/2)-1)./LR ,2));
f1 = h*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2));
f2 = h*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
f3 = h*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));
% Main time-stepping loop:
tt = 0;
tmax =25000; nmax = round(tmax/h); nplt = 1;%floor((tmax/10000)/h);
g = -0.5i*k;

vv = zeros(N, nmax);

vv(:,1) = v;

for n = 1:nmax
t = n*h;
Nv = g.*fft(real(ifft(v)).^2);
a = E2.*v + Q.*Nv;
Na = g.*fft(real(ifft(a)).^2);
b = E2.*v + Q.*Na;
Nb = g.*fft(real(ifft(b)).^2);
c = E2.*a + Q.*(2*Nb-Nv);
Nc = g.*fft(real(ifft(c)).^2);
% V_hat = fft2(v);
KV_hat=K_hat.*v;
% kv=ifft2(KV_hat);
v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3 + KK*(KV_hat-v)*h;
% v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3 + KK*(1+i*v)*(kv-v)*h;
vv(:,n) = v;
end

uu = transpose(real(ifft(vv)));

%%

% fig2 = figure('pos',[5 270 600 200],'color','w');
% imagesc(transpose(uu))
% shading flat
% colormap(jet);
% colorbar;

%%

%save KS_100.mat uu -v7.3;

% Adata = uu(1:80000,:);
% Bdata = uu(80001:end,:);
% 
% filename = strcat('Adata_L', num2str(d), '_N_', num2str(N), '_dps', num2str(80000), '.mat');
% save(filename, 'Adata', 'Bdata', '-v7.3');


train_input_sequence = uu(1:80000,:);

test_input_sequence = uu(80001:end,:);

figure(); surf(uu.'); view(0,90); shading interp; axis tight;

save('test_input_sequence_nonlocal.mat', 'test_input_sequence', '-v7.3');
save('train_input_sequence_nonlocal.mat', 'train_input_sequence', '-v7.3');

% save('test_input_sequence_L44.mat', 'test_input_sequence', '-v7.3');
% save('train_input_sequence_L44.mat', 'train_input_sequence', '-v7.3');

% save('test_input_sequence2.mat', 'test_input_sequence', '-v7.3');
% save('train_input_sequence2.mat', 'train_input_sequence', '-v7.3');
