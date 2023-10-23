
% Periodic boundary conditions are used.
clear all;
% disp('*** 1D NLCGL SIMULATION ***');
i_movei_index=1;
count_m=1;
drawing=1;
% Set equation coefficients
a = -2; %-2
b = 2;
c0= 0.002;
KK=1;
% save_results = false;
save_results = true;

% Set system parameters
% L    = 50;        % Domain size (assume square container) 50 20, default
% L    = 20;        % Domain size (assume square container) 50 20, default
L    = 18;        % Domain size (assume square container) 50 20, default
% L    = 10;        % Domain size (assume square container) 50 20
% L    = 5;        % Domain size (assume square container) 50 20
% L    = 2;        % Domain size (assume square container) 50 20
% L    = 1;        % Domain size (assume square container) 50 20
% Tmax = 80;        % End time, default
N    = 32;       % Number of grid points, default
% N    = 64;       % Number of grid points, default
% N    = 20;       % Number of grid points
% N    = 10;       % Number of grid points
% N    = 6;       % Number of grid points
% time_scale = 70;
% time_scale = 50;
% time_scale = 10;
time_scale = 1;
% dT   = 0.05;      % Timestep, default
% dT   = 0.05/time_scale;      % Timestep
dT   = 0.07;      % Timestep
sample_dT = 0.07;

type = 'train';
% type = 'test';
% dps  = 1000000;  %50 200     % Number of stored times
% dps  = 700000;  %50 200     % Number of stored times
% dps  = 300000;  %50 200     % Number of stored times
% dps  = 140000;  %50 200     % Number of stored times
% dps  = 100000;  %50 200     % Number of stored times
% dps  = 80000 + 5000;  %50 200     % Number of stored times
% dps  = 80000 + 10;  %50 200     % Number of stored times
dps  = 80000;  %50 200     % Number of stored times
% dps  = 60000;  %50 200     % Number of stored times
% dps  = 40000;  %50 200     % Number of stored times
% dps  = 30000;  %50 200     % Number of stored times
% dps  = 20001;  %50 200     % Number of stored times
% dps  = 20000;  %50 200     % Number of stored times
% dps  = 10000;  %50 200     % Number of stored times
% dps  = 2000;  %50 200     % Number of stored times
% dps  = 1000;  %50 200     % Number of stored times
% dps  = 800;  %50 200     % Number of stored times
% dps  = 400;  %50 200     % Number of stored times
% dps  = 200;  %50 100     % Number of stored times, default
% dps  = 100;  %50 100     % Number of stored times, default
Tmax = sample_dT / dT * dps * dT *2;
co   = 0;         % Whether to continue simulation from final values of previous

% Calculate some further parameters
nmax  = round(Tmax/dT);
XX    = (L/N)*(-N/2:N/2-1)'; 
% [X,Y] = meshgrid(XX);
X = XX;
nplt  = floor(nmax/dps);

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

% Define initial conditions
if co == 0 
	Tdata = zeros(1,dps+1);
	Tdata(1) = 0;
    initnoise = 10^-2;
    % initnoise = 10;
	% A = zeros(size(X)) + 10^(-2)*randn(size(X));
	% A = zeros(size(X)) + 1*randn(size(X));
	A = zeros(size(X)) + initnoise*randn(size(X));
%	A = complex(double(C)/256);
else
	% A         = Adata(:,:,end);
    A = Adata(:,end);
	starttime = Tdata(end);
	Tdata     = zeros(1,dps+1);
	Tdata(1)  = starttime;
	disp('    CARRYING OVER...')
end	

% Set wavenumbers.
k  				= [0:N/2-1 0 -N/2+1:-1]'*(2*pi/L);
k2 				= k.*k;  
k2(N/2+1) = ((N/2)*(2*pi/L))^2;
% [k2x,k2y] = meshgrid(k2); 
k2x = k2;
% del2 			= k2x+k2y;
del2 = k2x;
% Adata     = zeros(N,N,dps+1);
% A_hatdata = zeros(N,N,dps+1);
Adata = zeros(N, dps+1);
A_hatdata = zeros(N, dps+1);

A_hat            = fft2(A);
% Adata(:,:,1)     = A;
% A_hatdata(:,:,1) = A_hat;
Adata(:,1) = A;
A_hatdata(:,1) = A_hat;

% Compute exponentials and nonlinear factors for ETD2 method
%cA 	    	= 1+c0 - del2*(1+i*a);
%expA 	  	= exp(dT*cA);
%nlfacA  	= (exp(dT*cA).*(1+1./cA/dT)-1./cA/dT-2)./cA;
%nlfacAp 	= (exp(dT*cA).*(-1./cA/dT)+1./cA/dT+1)./cA;

% Solve PDE
dataindex = 2;

hwaitbar = waitbar(0,'Please wait...'); 
progress = 0; total = nmax;
hwaitbar = waitbar(progress/total,hwaitbar,... 
    sprintf('calculating prob: %d/%d', progress, total));
% hnlcgl = figure('position', [200 200 300 300]); %% delete
for n = 1:nmax
	T = Tdata(1) + n*dT;
	%A = ifft2(A_hat);
	A_hat            = fft2(A);
    
	% Find nonlinear component in Fourier space
	%nlA	= -(1+i*b)*fft2(A.*abs(A).^2);
	
	% Setting the first values of the previous nonlinear coefficients
	%if n == 1
	%	nlAp = nlA;
    %end
    
    KA_hat=K_hat.*A_hat;
    KA=ifft2(KA_hat);
    %nonlocal=KK*(1+i*a)*(KA_hat-A_hat); 
    

	% Time-stepping
	%A_hat = A_hat.*expA + nlfacA.*nlA + nlfacAp.*nlAp;
	%nlAp  = nlA; bn
	
    A=A+((1+i*c0)*A -(1+i*b)*(A.*abs(A).^2)+KK*(1+i*a)*(KA-A))*dT;
    
	% Saving data
	if mod(n,nplt) == 1 
	%	A = ifft2(A_hat);
		% Adata(:,:,dataindex)     = A; 
		% A_hatdata(:,:,dataindex) = A_hat; 
        Adata(:,dataindex) = A;
        A_hatdata(:,dataindex) = A_hat;

		Tdata(dataindex)         = T;
		dataindex                = dataindex + 1;
% Plot
       continue;
%        if drawing==1
%         % h1 = figure('position', [200 200 300 300]);
% %        figure(1);
% %        pcolor(X,Y,real(Adata(:,:,dataindex-1)))
% 
%         surf(X,Y, real(Adata(:,:,dataindex-1)));
% 
%         caxis([-1.2, 1.2]);
% %       colormap(cool);
%         view(0,90), shading interp, axis tight
%         mov(i_movei_index)=getframe(gcf);
%         i_movei_index=i_movei_index+1;
%         % clf;
%         % close(h1); %% modify
%        end;
	end
	

    progress = n;
    total = nmax;
    hwaitbar = waitbar(progress/total,hwaitbar,... 
    sprintf('conducting simulation: %d/%d', progress, total));
end
% close(hnlcgl); %% modify
close(hwaitbar);

data_dir = '';
% data_dir = 'results/CGL1d/';
%% save for parallel
switch type
    case 'train' % train data
        n_steps = dps;
        % data_kind = 'NLCGL';
        data_kind = 'NLCGL';
        % L = 8; N = 32; dps = n_steps;
        n_data = 2 * N;
        filename = [data_dir data_kind '_L' num2str(L), '_N_' num2str(N) '_dps' num2str(dps) 'c1_' num2str(a) 'c2_' num2str(b) '.mat'];
        Adata = [real(Adata(:,1:end-1)); imag(Adata(:,1:end-1))].';
        % Adata = Adata.';
        train_input_sequence = zeros(n_steps, n_data);
        train_input_sequence(:, 1:2:end) = Adata(:, 1:n_data/2);
        train_input_sequence(:, 2:2:end) = Adata(:, n_data/2+1:end);
        save(filename, 'train_input_sequence', '-v7.3');
    case 'test' % test data
        n_steps = dps;
        % data_kind = 'NLCGL';
        data_kind = 'NLCGL';
        % L = 8; N = 32; dps = n_steps;
        n_data = 2 * N;
        filename = [data_dir data_kind '_L' num2str(L), '_N_' num2str(N) '_dps' num2str(dps) 'c1_' num2str(a) 'c2_' num2str(b) '.mat'];
        Adata = [real(Adata(:,1:end-1)); imag(Adata(:,1:end-1))].';
        % Adata = Adata.';
        test_input_sequence = zeros(n_steps, n_data);
        test_input_sequence(:, 1:2:end) = Adata(:, 1:n_data/2);
        test_input_sequence(:, 2:2:end) = Adata(:, n_data/2+1:end);
        save(filename, 'test_input_sequence', '-v7.3');
end
if save_results
    figure();
    surf(real(Adata));
    view(0,90); shading interp; axis tight; colorbar;
%     savefig(strcat('results/NLCGL1d/NLCGL1d_', '_L', num2str(L), '_N_', num2str(N), '_dps', num2str(dps), '.fig'))
end

% if save_results
%     filename = strcat('results/NLCGL1d/Adata', '_L', num2str(L), '_N_', num2str(N), '_dps', num2str(dps), '.mat');
%     save(filename, 'Adata', '-v7.3');
%     save(strcat('results/NLCGL1d/NLCGL', '_L', num2str(L), '_N_', num2str(N), '_dps', num2str(dps), '.mat'), '-v7.3');
%     display(filename);
% end
