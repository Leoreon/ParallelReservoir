% �`���[�����O�p�^�[���̐����v���O����
% �����n�ɂ�FitzHugh-Nagumo ���������g�p
% �Q�l�����E�Q�l�T�C�g
% -. Turing Patterns by Anisotropic Diffusion, Iwamoto
% -. https://kmaeda.net/kmaeda/demo/rds-fn/, �O�c��M


% Setting
x_length=200;
y_length=200;
p = 1; % Orientation Dispersion(0<p<2): �Ȃ̕��������܂�
q = 1; % p=1,q=1.5�ł��������̎Ȗ͗l�ɂȂ�
d_u = 0.006; 
d_v = 0.07;
dt = 0.01;
a = 1.2; 
b = 0.5;
c = 0; % �����p�^�[����random�̏ꍇ�C0�ŎȖ͗l, �}0.2�Ŕ��͗l
epsilon = 0.1;

% Initial Pattern
% pattern = input('[1]target,[2]:line,[3]:random:');
pattern = 3;
if pattern~=1 && pattern~=2 && pattern~=3
    disp('bad input')
return
end
switch pattern
    case 1 % target
        U = zeros(x_length,y_length);
        V = zeros(x_length,y_length);
        for x = 1:x_length
            for y = 1:y_length
                if (x-x_length/2)^2+(y-y_length/2)^2 < 10
                    U(x,y)=1;
                end
            end
        end
    case 2 % line
        U = zeros(x_length,y_length);
        V = zeros(x_length,y_length);
        U(:,x_length/2-1:y_length/2-1)=1;
        
    case 3 % random
        U = rand(x_length,y_length)*2-1;
        V = rand(x_length,y_length)*2-1;
end

% Main
figure('Color','white')
for step = 1:10000
    dU=d_u*delta(U,x_length,y_length,epsilon,p)+U-U.^3-V;
    dV=d_v*delta(V,x_length,y_length,epsilon,q)+a*(U-b*V-c);
    U = dU * dt + U;
    V = dV * dt + V;
    if mod(step,100)==0
        image(V,'CDataMapping','scaled')
        pbaspect([1 1 1])
        colormap(gray)
        drawnow
    end
end

% Calc of Anisotropic Diffusion
function out = delta(M,x_length,y_length,epsilon,p)
out = zeros(x_length,y_length);
for x=1:x_length
    for y=1:y_length
        if x == 1
            tmp = p * M(x_length,y);
        else
            tmp = p * M(x-1,y);
        end
        if y == 1
            tmp = (2-p) * M(x,y_length) + tmp;
        else
            tmp = (2-p) * M(x,y-1) + tmp;
        end
        if x == x_length
            tmp = p * M(1,y) + tmp;
        else
            tmp = p * M(x+1,y) + tmp;
        end
        if y == y_length
            tmp = (2-p) * M(x,1) + tmp;
        else
            tmp = (2-p) * M(x,y+1) + tmp;
        end
        out(x,y) = (tmp - 4 * M(x,y))/(epsilon^2);
    end
end
end