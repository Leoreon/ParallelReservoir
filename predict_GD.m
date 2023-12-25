% function prediction = predict(w,w_out,x,w_in,pl,chunk_size,num_reservoirs_per_worker,frontWkrIdx, rearWkrIdx,N, locality, num_kind_data, test_in)
function [prediction, delta_list] = predict_GD(L, w,w_out,x,w_in,pl,chunk_size,num_reservoirs_per_worker,frontWkrIdx, rearWkrIdx,N, rear_locality_data, forward_locality_data, num_kind_data, test_in)
% display('start predict');
prediction = zeros(num_reservoirs_per_worker*chunk_size,pl);
delta_list = zeros(1, pl-1);
num_workers = numlabs;
l = labindex;
inputExist = ~isempty(test_in); % inputがないなら0、あるなら1

%% calculate time-derivative of KS
n_input = num_workers * chunk_size;
ks = [0:n_input/2-1 0 -n_input/2+1:-1]'*(2*pi/L); % wave numbers
h = 1/4;
Ls = ks.^2 - ks.^4; % Fourier multipliers
E = exp(h*Ls); E2 = exp(h*Ls/2);
M = 16; % no. of points for complex means
r = exp(1i*pi*((1:M)-.5)/M); % roots of unity

LR = h*Ls(:,ones(M,1)) + r(ones(n_input,1),:);


Q = h*real(mean( (exp(LR/2)-1)./LR ,2));
f1 = h*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2));
f2 = h*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
f3 = h*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));
% Main time-stepping loop:
% tt = 0;
% nmax = round(tmax/h); nplt = 1;%floor((tmax/10000)/h);
% g = -0.5i*k; % sample code (Pathak)
g = -0.5i*ks; % sample code (Pathak)
% g = 0.5i*k; % science of synchronization (kuramoto)


% if locality > 2 * chunk_size
if max(rear_locality_data, forward_locality_data) > 2 * chunk_size
    secFrontWkrIdx = mod(frontWkrIdx, numlabs) + 1;
    secRearWkrIdx = mod(rearWkrIdx-2, numlabs) + 1;

    thiFrontWkrIdx = mod(frontWkrIdx+1, numlabs) + 1;
    thiRearWkrIdx = mod(rearWkrIdx-3, numlabs) + 1;
    % display(frontWkrIdx);
    % display(rearWkrIdx);
    % display(secFrontWkrIdx);
    % display(secRearWkrIdx);
    for i=1:pl
        x_augment = x;
        x_augment(2:2:N) = x_augment(2:2:N).^2;
        out = (w_out)*x_augment;
        if num_kind_data == 6
            for k = 1:size(out, 1)/num_kind_data
                nxyz = out((k-1)*num_kind_data+2:(k-1)*num_kind_data+4);
                norm_ = norm(nxyz);
                out((k-1)*num_kind_data+2:(k-1)*num_kind_data+4) = nxyz / 2 / norm_;
            end
        end
        labBarrier;
        rear_out = labSendReceive(frontWkrIdx, rearWkrIdx, out(end-chunk_size+1:end));
        front_out = labSendReceive(rearWkrIdx, frontWkrIdx, out(1:chunk_size));
        labBarrier;
        second_rear_out = labSendReceive(secFrontWkrIdx, secRearWkrIdx, out(end-chunk_size+1:end));
        second_front_out = labSendReceive(secRearWkrIdx, secFrontWkrIdx, out(1:chunk_size));
        labBarrier;
        % third_rear_out = labSendReceive(thiFrontWkrIdx, thiRearWkrIdx, out(end-(locality*num_kind_data-2*chunk_size)+1:end));
        third_rear_out = labSendReceive(thiFrontWkrIdx, thiRearWkrIdx, out(end-(rear_locality_data-2*chunk_size)+1:end));
        % third_front_out = labSendReceive(thiRearWkrIdx, thiFrontWkrIdx, out(1:(locality*num_kind_data-2*chunk_size)));
        third_front_out = labSendReceive(thiRearWkrIdx, thiFrontWkrIdx, out(1:(forward_locality_data-2*chunk_size)));
        % fprintf('out: \n');
        % display(size(out));
        % display(size(front_out));
        % display(size(rear_out));
        % break;
        feedback = vertcat(third_rear_out, second_rear_out, rear_out, out, front_out, second_front_out, third_front_out, test_in(:, (i):(i)*inputExist));
        % display(size(feedback));
        x = tanh(w*x + w_in*feedback); 
        prediction(:,i) = out;
        if i > 1
            switch l
                case 1
                    current_u = zeros(num_workers*chunk_size, 1);
                    next_u = zeros(num_workers*chunk_size, 1);
                    current_u(1:chunk_size) = prediction(:, i-1);
                    next_u(1:chunk_size) = prediction(:, i);
                    labBarrier;
                    for k = 2:numlabs
                        current_u(chunk_size*(k-1)+1:chunk_size*k) = labReceive(k);
                        next_u(chunk_size*(k-1)+1:chunk_size*k) = labReceive(k);
                    end
                    dt = 1/4;
                    dudt_res = (next_u - current_u)/dt;
                    dudt_KS = dKS(current_u);
                    delta = norm(dudt_res - dudt_KS)/n_input;
                    if i==30
                        % display(current_u);
                        % display(dudt_res - dudt_KS);
                        % display(dudt_res(1:5));
                        % display(dudt_KS(1:5));
                        
                        % display(delta);
                    end
                    delta_list(1, i-1) = delta;
                otherwise
                    current_u_ = prediction(:, i-1);
                    next_u_ = prediction(:, i);
                    labBarrier;
                    labSend(current_u_, 1);
                    labSend(next_u_, 1);
            end
        end
    end
% elseif locality > chunk_size
elseif max(rear_locality_data, forward_locality_data) > chunk_size
    secFrontWkrIdx = mod(frontWkrIdx, numlabs) + 1;
    secRearWkrIdx = mod(rearWkrIdx-2, numlabs) + 1;
    % display(frontWkrIdx);
    % display(rearWkrIdx);
    % display(secFrontWkrIdx);
    % display(secRearWkrIdx);
    for i=1:pl
        x_augment = x;
        x_augment(2:2:N) = x_augment(2:2:N).^2;
        out = (w_out)*x_augment;
        if num_kind_data == 6
            for k = 1:size(out, 1)/num_kind_data
                nxyz = out((k-1)*num_kind_data+2:(k-1)*num_kind_data+4);
                norm_ = norm(nxyz);
                out((k-1)*num_kind_data+2:(k-1)*num_kind_data+4) = nxyz / 2 / norm_;
            end
        end
        labBarrier;
        rear_out = labSendReceive(frontWkrIdx, rearWkrIdx, out(end-chunk_size+1:end));
        front_out = labSendReceive(rearWkrIdx, frontWkrIdx, out(1:chunk_size));
        labBarrier;
        % second_rear_out = labSendReceive(secFrontWkrIdx, secRearWkrIdx, out(end-(locality*num_kind_data-chunk_size)+1:end));
        second_rear_out = labSendReceive(secFrontWkrIdx, secRearWkrIdx, out(end-(rear_locality_data-chunk_size)+1:end));
        % second_front_out = labSendReceive(secRearWkrIdx, secFrontWkrIdx, out(1:(locality*num_kind_data-chunk_size)));
        second_front_out = labSendReceive(secRearWkrIdx, secFrontWkrIdx, out(1:(forward_locality_data-chunk_size)));
        % fprintf('out: \n');
        % display(size(out));
        % display(size(front_out));
        % display(size(rear_out));
        % break;
        feedback = vertcat(second_rear_out, rear_out, out, front_out, second_front_out, test_in(:, (i):(i)*inputExist));
        % display(size(feedback));
        x = tanh(w*x + w_in*feedback); 
        prediction(:,i) = out;
        if i > 1
            switch l
                case 1
                    current_u = zeros(num_workers*chunk_size, 1);
                    next_u = zeros(num_workers*chunk_size, 1);
                    current_u(1:chunk_size) = prediction(:, i-1);
                    next_u(1:chunk_size) = prediction(:, i);
                    labBarrier;
                    for k = 2:numlabs
                        current_u(chunk_size*(k-1)+1:chunk_size*k) = labReceive(k);
                        next_u(chunk_size*(k-1)+1:chunk_size*k) = labReceive(k);
                    end
                    dt = 1/4;
                    dudt_res = (next_u - current_u)/dt;
                    dudt_KS = dKS(current_u);
                    delta = norm(dudt_res - dudt_KS)/n_input;
                    if i==30
                        % display(current_u);
                        % display(dudt_res - dudt_KS);
                        % display(dudt_res(1:5));
                        % display(dudt_KS(1:5));
                        
                        % display(delta);
                    end
                    delta_list(1, i-1) = delta;
                otherwise
                    current_u_ = prediction(:, i-1);
                    next_u_ = prediction(:, i);
                    labBarrier;
                    labSend(current_u_, 1);
                    labSend(next_u_, 1);
            end
        end
    end
else
    for i=1:pl
        x_augment = x;
        x_augment(2:2:N) = x_augment(2:2:N).^2;
        % display('ee');
        out = (w_out)*x_augment;
        % display(num_kind_data);
        if num_kind_data == 6
            for k = 1:size(out, 1)/num_kind_data
                nxyz = out((k-1)*num_kind_data+2:(k-1)*num_kind_data+4);
                norm_ = norm(nxyz);
                out((k-1)*num_kind_data+2:(k-1)*num_kind_data+4) = nxyz / 2 / norm_;
            end
        end
        % display(norm(out(2:4)));
        % display('ff');
        % if numlabs ~= 1
        if ~(rear_locality_data == 0 || forward_locality_data == 0)
            labBarrier;
            % % if rear_locality_data ~= 0
            %     % rear_out = labSendReceive(frontWkrIdx ,rearWkrIdx, out(end-locality*num_kind_data+1:end));
            % % rear_out = labSendReceive(frontWkrIdx ,rearWkrIdx, out(end-rear_locality_data+1:end));
            % front_out = labSendReceive(frontWkrIdx ,frontWkrIdx, out(end-rear_locality_data+1:end));
            % % end
            % % if forward_locality_data ~= 0
            %     % front_out = labSendReceive(rearWkrIdx, frontWkrIdx, out(1:locality*num_kind_data));
            % % front_out = labSendReceive(rearWkrIdx, frontWkrIdx, out(1:forward_locality_data));
            % rear_out = labSendReceive(rearWkrIdx, rearWkrIdx, out(1:forward_locality_data));
            % % end
            
            % front_out = labSendReceive(frontWkrIdx ,frontWkrIdx, out(end-rear_locality_data+1:end));
            % labSend(out(end-forward_locality_data+1:end), frontWkrIdx);
            % rear_out = labReceive(rearWkrIdx);
            rear_out = labSendReceive(frontWkrIdx, rearWkrIdx, out(end-forward_locality_data+1:end));
    
            % labSend(out(1:rear_locality_data), rearWkrIdx);
            % front_out = labReceive(frontWkrIdx);
            front_out = labSendReceive(rearWkrIdx, frontWkrIdx, out(1:rear_locality_data));
            
            % fprintf('out: \n');
            % display(size(rear_out));
            % display(size(out));
            % display(size(front_out));
            % break;
            feedback = vertcat(rear_out, out, front_out, test_in(:, i:(i)*inputExist));
        else
            feedback = vertcat(out, test_in(:, i:(i)*inputExist));
        end
        % display(size(test_in));
        % feedback = vertcat(rear_out, out, front_out);
        % display(size(w));
        % display(size(x));
        % display(size(w_in));
        % display(size(feedback));
        % display(size(rear_out));
        % display(size(out));
        % display(size(front_out));
        x = tanh(w*x + w_in*feedback); 
        prediction(:,i) = out;
        if i > 1
            switch l
                case 1
                    current_u = zeros(num_workers*chunk_size, 1);
                    next_u = zeros(num_workers*chunk_size, 1);
                    current_u(1:chunk_size) = prediction(:, i-1);
                    next_u(1:chunk_size) = prediction(:, i);
                    labBarrier;
                    for k = 2:numlabs
                        current_u(chunk_size*(k-1)+1:chunk_size*k) = labReceive(k);
                        next_u(chunk_size*(k-1)+1:chunk_size*k) = labReceive(k);
                    end
                    dt = 1/4;
                    dudt_res = (next_u - current_u)/dt;
                    dudt_KS = dKS(current_u);
                    delta = norm(dudt_res - dudt_KS)/n_input;
                    if i==30
                        % display(current_u);
                        % display(dudt_res - dudt_KS);
                        % display(dudt_res(1:5));
                        % display(dudt_KS(1:5));
                        
                        % display(delta);
                    end
                    delta_list(1, i-1) = delta;
                otherwise
                    current_u_ = prediction(:, i-1);
                    next_u_ = prediction(:, i);
                    labBarrier;
                    labSend(current_u_, 1);
                    labSend(next_u_, 1);
            end
        end
    end
end

function dudt = dKS(ut)
% u_hat = fftshift(fft(ut));
% u2_hat = fftshift(fft(ut.^2));
v = (fft(ut));
% u2_hat = (fft(ut.^2));

Nv = g.*fft(real(ifft(v)).^2);
a = E2.*v + Q.*Nv;
Na = g.*fft(real(ifft(a)).^2);
b = E2.*v + Q.*Na;
Nb = g.*fft(real(ifft(b)).^2);
c = E2.*a + Q.*(2*Nb-Nv);
Nc = g.*fft(real(ifft(c)).^2);
next_v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3; % E.*v : term of y*dy/dx
dudt = (ifft(next_v) - ifft(v)) / h;
% display(size(u2_hat));
% display(size(ks));
% ks = 1:length(ut);
% du_hatdt = -j*ks/2.*u2_hat + Ls .* u_hat;
% dudt = ifft((du_hatdt));
% dudt = ifft(ifftshift(du_hatdt));
end

function norm_ = norm(vec)
    norm_ = sqrt(sum(vec.*vec));
end
end
