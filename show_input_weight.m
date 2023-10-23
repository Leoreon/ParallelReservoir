

weight = [zeros(1, 5) 0:1/5:1 ones(1, 10-1) 1:-1/5:0 zeros(1, 5)];
n = size(weight, 2);
index = linspace(0, n-1, n);
figure(); plot(index, weight);
xlim([0 n]); ylim([-0 1.2]); 
xlabel('index'); ylabel('input weight');
fontsize(16, 'points');


weight = [zeros(1, 50) ones(1, 60) ones(1, 100-1) ones(1, 60) zeros(1, 50)];
n = size(weight, 2);
index = linspace(0, n-1, n)/10;
figure(); plot(index, weight);
xlim([0 n/10]); ylim([-0 1.2]); 
xlabel('index'); ylabel('d');
fontsize(16, 'points');
