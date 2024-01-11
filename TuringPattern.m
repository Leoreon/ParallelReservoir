% FitzHugh-Nagumoモデルのパラメータ
alpha = 0.2;
beta = 0.2;
gamma = 3;
% 空間格子のサイズと時間ステップ
L = 40; % 空間のサイズ
Nx = 100; % 格子の点の数
dx = L / Nx; % 空間ステップ
dt = 0.01; % 時間ステップ
% 初期条件の設定
u = zeros(Nx, Nx);
v = zeros(Nx, Nx);
% 中央に初期の励起を与える
u(Nx/2, Nx/2) = 0.5;
v(Nx/2, Nx/2) = 0.25;
% 反応拡散方程式の数値解法
for t = 1:1000
    % 反応項
    du = alpha * (u - u.^3 / 3 - v + gamma);
    dv = beta * (u - v);
    % ラプラシアンの計算
    laplacian_u = del2(u, dx);
    laplacian_v = del2(v, dx);
    % 反応拡散方程式の更新
    u = u + dt * (laplacian_u + du);
    v = v + dt * (laplacian_v + dv);
    % 一定の間隔でパターンをプロット
    if mod(t, 50) == 0
        figure;
        subplot(1, 2, 1);
        imagesc(u);
        title(['u at t = ', num2str(t * dt)]);
        axis square;
        colormap jet;
        subplot(1, 2, 2);
        imagesc(v);
        title(['v at t = ', num2str(t * dt)]);
        axis square;
        colormap jet;
        drawnow;
    end
end














