% ローレンツ方程式の定義
lorenz = @(t, y) [10 * (y(2) - y(1));
                  28 * y(1) - y(2) - y(1) * y(3);
                  y(1) * y(2) - 8/3 * y(3)];
% 初期条件
initial_conditions = [1; 0; 0];
% 時間範囲
tspan = [0 50];
% 方程式を数値的に解く
[t, sol] = ode45(lorenz, tspan, initial_conditions);
% 結果をプロット
figure;
plot3(sol(:, 1), sol(:, 2), sol(:, 3));
xlabel('x');
ylabel('y');
zlabel('z');
view(45, 25); fontsize(16, 'points');
% title('ローレンツ方程式の解');
% 軸を等しいスケールに設定
axis equal;
% グリッドを表示
grid on;

% 初期条件
initial_conditions2 = [1+1e-1; 0; 0];
% 時間範囲
tspan2 = [0 50];
% 方程式を数値的に解く
[t2, sol2] = ode45(lorenz, tspan2, initial_conditions2);
% 結果をプロット
figure;
plot3(sol2(:, 1), sol2(:, 2), sol2(:, 3));
xlabel('x');
ylabel('y');
zlabel('z');
view(45, 25); fontsize(16, 'points');
% title('ローレンツ方程式の解');
% 軸を等しいスケールに設定
axis equal;
% グリッドを表示
grid on;
n = 1000;
figure;
plot3(sol(1:n, 1), sol(1:n, 2), sol(1:n, 3));
xlabel('x');
ylabel('y');
zlabel('z');
view(45, 25); fontsize(16, 'points');
% title('ローレンツ方程式の解');
% 軸を等しいスケールに設定
axis equal;
% グリッドを表示
grid on
hold on;
plot3(sol2(1:n, 1), sol2(1:n, 2), sol2(1:n, 3));

error = sqrt(sum((sol(1:n,:)-sol2(1:n,:)).^2, 2));

figure(); plot(error);
xlabel('time'); ylabel('error'); fontsize(16, 'points');


