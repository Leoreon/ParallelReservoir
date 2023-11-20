

train_dir = 'LCD_data/data/train';
test_dir = 'LCD_data/data/test';
% % nx = load(strcat(test_dir, 'nx.dat'));
%% load data of Liquid Crystal Display
train_filename = [train_dir 'LCD_data.mat'];
test_filename = [test_dir 'LCD_data.mat'];
m = load(train_filename);
tf = load(test_filename);

% % Ex.dat: 電解のx成分
% % Ez.dat: 電解のz成分
% % nx.dat: 液晶の向きのｘ成分
% % ny.dat: 液晶の向きのｙ成分
% % nz.dat: 液晶の向きのｚ成分
% % pot.dat: 電位
% % vol_pix.dat: 画素電極の電位
% 
% Ez = importdata(fullfile(train_dir, 'elec.dat')).data;
% % Ex = importdata(fullfile(test_dir, 'Ex.dat')).data;
% % Ez = importdata(fullfile(test_dir, 'Ez.dat')).data;
% nx = importdata(fullfile(train_dir, 'nx.dat')).data;
% ny = importdata(fullfile(train_dir, 'ny.dat')).data;
% nz = importdata(fullfile(train_dir, 'nz.dat')).data;
% ux = importdata(fullfile(train_dir, 'ux.dat')).data;
% uy = importdata(fullfile(train_dir, 'uy.dat')).data;
% % pot = importdata(fullfile(test_dir, 'pot.dat')).data;
% vol_p = importdata(fullfile(train_dir, 'volp.dat')).data;
% % elec = importdata(fullfile(test_dir, 'elec.dat')).data;
% 
% 
% times = reshape(Ez(:,1), 100, []);
% xs = reshape(Ez(:,2), 100, []);
% 
% Ezs = reshape(Ez(:,3), 100, []);
% % figure(); surf(times, xs, Ezs);
% % view(0, 90); shading interp, axis tight; xlabel('time (step)'); ylabel('location'); title('electric field');
% % colorbar;
% 
% nxs = reshape(nx(:,3), 100, []);
% % figure(); surf(times, xs, nxs);
% % view(0, 90); shading interp, axis tight; xlabel('time (step)'); ylabel('location'); title('orientation along x axis');
% % colorbar;
% 
% nys = reshape(ny(:,3), 100, []);
% % figure(); surf(times, xs, nys);
% % view(0, 90); shading interp, axis tight; xlabel('time (step)'); ylabel('location'); title('orientation along y axis');
% % colorbar;
% 
% nzs = reshape(nx(:,3), 100, []);
% % figure(); surf(times, xs, nzs);
% % view(0, 90); shading interp, axis tight; xlabel('time (step)'); ylabel('location'); title('orientation along z axis');
% % colorbar;
% 
% uxs = reshape(ux(:,3), 100, []);
% % figure(); surf(times, xs, uxs);
% % view(0, 90); shading interp, axis tight; xlabel('time (step)'); ylabel('location'); title('ux');
% % colorbar;
% 
% uys = reshape(uy(:,3), 100, []);
% % figure(); surf(times, xs, uys);
% % view(0, 90); shading interp, axis tight; xlabel('time (step)'); ylabel('location'); title('uy');
% % colorbar;
% 
% vols = reshape(vol_p(:,3), 100, []);
% % figure(); surf(times, xs, vols);
% % view(0, 90); shading interp, axis tight; xlabel('time (step)'); ylabel('location'); title('voltage');
% % colorbar;
% 
% 
% %% normalize
% Ezs = Ezs / max(max(abs(Ezs))) / 2;
% nxs = nxs / 2;
% nys = nys / 2;
% nzs = nzs / 2;
% uxs = uxs / max(max(abs(uxs))) / 2;
% uys = uys / max(max(abs(uys))) / 2;
% vols = vols / max(max(abs(vols))) / 2;
% 
% sample_dT = 5e-4;
% L = max(max(xs));
% Adata = [Ezs; nxs; nys; nzs; uxs; uys];
% Input = vols;
% x_ = xs(:, 1);
% time_ = times(1, :);
% 
% % clearvars -except Adata Input sample_dT;
% % filename = strcat('LCD1d/LCD_data.mat');
% % save(filename, 'Adata', 'Input', 'x_', 'time_', 'sample_dT', 'L');
% 
% % figure(); plot(times(1,:), nxs(10,:));
% 
