% CGLsim1D.m
% Copyright David M. Winterbottom 2005

% ************************************************************
% Simulating the complex Ginzburg-Landau equation in 1D using
% pseudo-spectral code and ETD2 exponential time-stepping (See
% Matthews and Cox, J. Comp. Phys., 176 430-455, 2002)
% ************************************************************
clear all;

disp('*** 1D CGL SIMULATION ***');

% Set equation coefficients
% a = -2;
% b = 0.7;
a = -1; b = 2;
% a = -2; b = 2;
% a = 2;% b = -2;

% a = -1.05; b = 1.05;
% a = -4; b = 1;
% a = 0; b = -3;

% save_results = false;
save_results = true;

% Set system parameters
% L    = 2000;       % Domain size
% L    = 200;       % Domain size
% L    = 50;       % Domain size
% L    = 44;       % Domain size
% L    = 40;       % Domain size
% L    = 36;       % Domain size
% L    = 26;       % Domain size
L    = 22;       % Domain size
% L    = 18;       % Domain size
% L    = 14;       % Domain size
% L    = 8;       % Domain size
% Tmax = 200;       % Simulation time
% N    = 512;       % Number of grid points
% dT   = 0.05;      % Timestep (choose between 0.01 and 0.05)
% dps  = 1000;      % Number of stored times
% ic   = 'uniform';   % Initial condition: choose 'zero', 'tw', 'uniform' or 'pulse'
ic   = 'pulse';   % Initial condition: choose 'zero', 'tw', 'uniform' or 'pulse'
n    = 0;         % Winding number for 'tw' initial condition
% N    = 2048;       % Number of grid points, default
% N    = 256;       % Number of grid points, default
% N    = 128;       % Number of grid points, default
% N    = 64;       % Number of grid points, default
N    = 32;       % Number of grid points, default
% N    = 20;       % Number of grid points
% N    = 10;       % Number of grid points
% N    = 6;       % Number of grid points
% time_scale = 70;
% time_scale = 50;
% time_scale = 10;
time_scale = 1;
% dT   = 0.05;      % Timestep, default
% dT   = 0.05/time_scale;      % Timestep
% dT   = 0.07;      % Timestep
dT   = 0.01;      % Timestep
% dT   = 1e-4;      % Timestep
% sample_dT = 0.01;
sample_dT = 0.07;
% sample_dT = 0.1;
% sample_dT = 1e-4;

type = 'train';
% type = 'test';
% type = 'train&test';
% dps  = 1000000;  %50 200     % Number of stored times
% dps  = 700000;  %50 200     % Number of stored times
% dps  = 300000;  %50 200     % Number of stored times
% dps  = 140000;  %50 200     % Number of stored times
% dps  = 100000;  %50 200     % Number of stored times
% dps  = 80000 + 5000;  %50 200     % Number of stored times
% dps  = 80000 + 10;  %50 200     % Number of stored times
% dps  = 80000;  %50 200     % Number of stored times
% dps  = 60000;  %50 200     % Number of stored times
% dps  = 40000;  %50 200     % Number of stored times
% dps  = 30000;  %50 200     % Number of stored times
% dps  = 20001;  %50 200     % Number of stored times
% dps  = 20000;  %50 200     % Number of stored times
% dps  = 10000;  %50 200     % Number of stored times
% dps  = 2000;  %50 200     % Number of stored times
dps  = 1000;  %50 200     % Number of stored times
% dps  = 800;  %50 200     % Number of stored times
% dps  = 400;  %50 200     % Number of stored times
% dps  = 200;  %50 100     % Number of stored times, default
% dps  = 100;  %50 100     % Number of stored times, default
Tmax = 2 * sample_dT / dT * dps * dT;
co   = 0;         % Whether to continue simulation from final values of previous

% Calculate some further parameters
% nmax = round(Tmax/dT);
nmax = round(Tmax/dT);
q    = n*2*pi/L; 
X    = (L/N)*(-N/2:N/2-1)'; 
nplt = floor(nmax/dps);

% lyapunov_sum = 0;

% Define initial conditions
if co == 0 
	Tdata = zeros(1,dps+1);
	if strcmp(ic, 'zero')
		A = zeros(size(X)) + 10^(-2)*randn(size(X));
	elseif strcmp(ic, 'tw')
		A 	= sqrt(1-q^2)*exp(i*q*X) + 10^(-2)*randn(size(X));
    elseif strcmp(ic, 'uniform')
		% A = 2 * ones(size(X)) + 0.01*randn(size(X));
		A = ones(size(X)) + 0.01*randn(size(X));
    elseif strcmp(ic, 'pulse')
		A = sech((X+10).^2) + 0.8*sech((X-30).^2) + 10^(-2)*randn(size(X));
		% A = sech((X+10).^2) + 0.8*sech((X-30).^2) + 10^(-2)*randn(size(X));
	    A = 10 * A;
    else
		error('invalid initial condition selected')
	end
	Tdata(1) = 0;
else
	A         = Adata(:,end);
	starttime = Tdata(end);
	Tdata     = zeros(1,dps+1);
	Tdata(1)  = starttime;
	disp('    CARRYING OVER...')
end	

% Set wavenumbers and data arrays
k = [0:N/2-1 0 -N/2+1:-1]'*(2*pi/L);
k2 = k.*k; k2(N/2+1) = ((N/2)*(2*pi/L))^2;
Adata     = zeros(N,dps+1);
A_hatdata = zeros(N,dps+1);
A_hat          = fft(A);
Adata(:,1)     = A;
A_hatdata(:,1) = A_hat;

% Compute exponentials and nonlinear factors for ETD2 method
cA 	    	= 1-k2*(1+i*a);
expA 	  	= exp(dT*cA);
nlfacA  	= (exp(dT*cA).*(1+1./cA/dT)-1./cA/dT-2)./cA;
nlfacAp 	= (exp(dT*cA).*(-1./cA/dT)+1./cA/dT+1)./cA;

% Solve PDE
dataindex = 2;
for n = 1:nmax
	T = Tdata(1) + n*dT;
	A = ifft(A_hat);
	
	% Find nonlinear component in Fourier space
	nlA	= -(1+i*b)*fft(A.*abs(A).^2);
	
	% Setting the first values of the previous nonlinear coefficients
	if n == 1
		nlAp = nlA;
	end
	
	% Time-stepping
	A_hat = A_hat.*expA + nlfacA.*nlA + nlfacAp.*nlAp;
	nlAp  = nlA;
	
	% Saving data
	if mod(n,nplt) == 0 
        % nextA = ifft(A_hat);
        % ln = log(abs((nextA-A)/sample_dT));
        % prevA = Adata(:, int32(n/nplt));
        % ln = log(abs((A-prevA)/sample_dT));
        % if ln(1) == -Inf
        %     display(ln(1));
        %     break;
        % end
        % lyapunov_sum = lyapunov_sum + ln;

		A = ifft(A_hat);
		Adata(:,dataindex)     = A;
		A_hatdata(:,dataindex) = A_hat; 
		Tdata(dataindex)       = T;
		dataindex              = dataindex + 1;
    end

    % if n>10000
    %     nextA = ifft(A_hat);
    %     ln = log(abs((nextA-A)/dT));
    %     if ln(1) == -Inf
    %         display(ln(1));
    %         break;
    %     end
    %     lyapunov_sum = lyapunov_sum + ln;
    % end
	
	% Commenting on time elapsed
	if mod(n,floor(nmax/10)) == 0
		outp = strcat('  n= ', num2str(n), ' completed'); disp(outp);
	end
end
% lyapunov = lyapunov_sum / n * (sample_dT/dT);
% display(lyapunov);

% Plot evolution
figure('position', [200 200 300 350])
surf(X,Tdata,real(Adata).')
colorbar;
view(0,90), shading interp, axis tight;
% set(gca,'position', [0 0 1 1])
 set(gca,'position', [0.1 0.07 0.8 0.87])
set(gca,...
	'xcolor',		[0.6 0.6 0.6],...
	'ycolor',		[0.6 0.6 0.6],...
	'fontsize',	6,...
	'fontname', 'courier')
xlabel('X',...
	'fontname', 'courier',...
	'fontsize', 6,...
	'color',		[0.6 0.6 0.6])
ylabel('T',...
	'fontname', 'courier',...
	'fontsize', 6,...
	'color',		[0.6 0.6 0.6],...
	'rotation', 0)
title('evolution of |A|',...
	'fontname', 'courier',...
	'fontsize', 6,...
	'color', 		[0.6 0.6 0.6])

data_dir = '';
% data_dir = 'results/CGL1d/';
%% save for parallel
switch type
    case 'train' % train data
        n_steps = dps;
        % data_kind = 'NLCGL';
        data_kind = 'CGL';
        % L = 8; N = 32; dps = n_steps;
        n_data = 2 * N;
        filename = [data_dir 'CGL' '_L' num2str(L), '_N_' num2str(N) '_dps' num2str(dps) 'c1_' num2str(a) 'c2_' num2str(b) '.mat'];
        Adata = [real(Adata(:,1:end-1)); imag(Adata(:,1:end-1))].';
        % Adata = Adata.';
        train_input_sequence = zeros(n_steps, n_data);
        train_input_sequence(:, 1:2:end) = Adata(:, 1:n_data/2);
        train_input_sequence(:, 2:2:end) = Adata(:, n_data/2+1:end);
        save(filename, 'train_input_sequence', '-v7.3');
    case 'test' % test data
        n_steps = dps;
        % data_kind = 'NLCGL';
        data_kind = 'CGL';
        % L = 8; N = 32; dps = n_steps;
        n_data = 2 * N;
        filename = [data_dir 'CGL' '_L' num2str(L), '_N_' num2str(N) '_dps' num2str(dps) 'c1_' num2str(a) 'c2_' num2str(b) '.mat'];
        Adata = [real(Adata(:,1:end-1)); imag(Adata(:,1:end-1))].';
        % Adata = Adata.';
        test_input_sequence = zeros(n_steps, n_data);
        test_input_sequence(:, 1:2:end) = Adata(:, 1:n_data/2);
        test_input_sequence(:, 2:2:end) = Adata(:, n_data/2+1:end);
        save(filename, 'test_input_sequence', '-v7.3');
    case 'train&test'
        train_steps = 0.8 * dps;
        % data_kind = 'NLCGL';
        data_kind = 'CGL';
        % L = 8; N = 32; dps = n_steps;
        n_data = 2 * N;
        filename = [data_dir 'CGL' '_L' num2str(L), '_N_' num2str(N) '_dps' num2str(train_steps) 'c1_' num2str(a) 'c2_' num2str(b) '.mat'];
        Adata = [real(Adata(:,1:end-1)); imag(Adata(:,1:end-1))].';
        % Adata = Adata.';
        train_input_sequence = zeros(train_steps, n_data);
        train_input_sequence(:, 1:2:end) = Adata(1:train_steps, 1:n_data/2);
        train_input_sequence(:, 2:2:end) = Adata(1:train_steps, n_data/2+1:end);
        save(filename, 'train_input_sequence', '-v7.3');

        test_steps = 0.2 * dps;
        % data_kind = 'NLCGL';
        data_kind = 'CGL';
        % L = 8; N = 32; dps = n_steps;
        n_data = 2 * N;
        filename = [data_dir 'CGL_test' '_L' num2str(L), '_N_' num2str(N) '_dps' num2str(test_steps) 'c1_' num2str(a) 'c2_' num2str(b) '.mat'];
        % Adata = [real(Adata(:,1:end-1)); imag(Adata(:,1:end-1))].';
        % Adata = Adata.';
        test_input_sequence = zeros(test_steps, n_data);
        test_input_sequence(:, 1:2:end) = Adata(train_steps+1:train_steps+test_steps, 1:n_data/2);
        test_input_sequence(:, 2:2:end) = Adata(train_steps+1:train_steps+test_steps, n_data/2+1:end);
        save(filename, 'test_input_sequence', '-v7.3');
end

%% save for simulation_future
if false %save_results
    % filename = strcat('results/CGL1d/Adata', '_L', num2str(L), '_N_', num2str(N), '_dps', num2str(dps), '.mat');
    filename = strcat(data_dir, 'Adata', '_L', num2str(L), '_N_', num2str(N), '_dps', num2str(dps), 'c1_', num2str(a), 'c2_', num2str(b), '.mat');
    save(filename, 'Adata', '-v7.3');
    % filename2 = strcat('results/CGL1d/CGL', '_L', num2str(L), '_N_', num2str(N), '_dps', num2str(dps), '.mat');
    filename2 = strcat(data_dir, 'CGL', '_L', num2str(L), '_N_', num2str(N), '_dps', num2str(dps), 'c1_', num2str(a), 'c2_', num2str(b), '.mat');
    save(filename2, '-v7.3');
    display(filename);
end
