%% clear all variables
clear all

test_dir = 'LCD_data/data/test';
train_dir = 'LCD_data/data/train';
% nx = load(strcat(test_dir, 'nx.dat'));

%% load data of ekisho
% for data_dir = [train_dir test_dir]
data_dir = train_dir;
    % Ex.dat: 電解のx成分
    % Ez.dat: 電解のz成分
    % nx.dat: 液晶の向きのｘ成分
    % ny.dat: 液晶の向きのｙ成分
    % nz.dat: 液晶の向きのｚ成分
    % pot.dat: 電位
    % vol_pix.dat: 画素電極の電位
    
    Ez = importdata(fullfile(data_dir, 'elec.dat')).data;
    % Ex = importdata(fullfile(test_dir, 'Ex.dat')).data;
    % Ez = importdata(fullfile(test_dir, 'Ez.dat')).data;
    nx = importdata(fullfile(data_dir, 'nx.dat')).data;
    ny = importdata(fullfile(data_dir, 'ny.dat')).data;
    nz = importdata(fullfile(data_dir, 'nz.dat')).data;
    ux = importdata(fullfile(data_dir, 'ux.dat')).data;
    uy = importdata(fullfile(data_dir, 'uy.dat')).data;
    % pot = importdata(fullfile(test_dir, 'pot.dat')).data;
    vol_p = importdata(fullfile(data_dir, 'volp.dat')).data;
    % elec = importdata(fullfile(test_dir, 'elec.dat')).data;
    
    
    times = reshape(Ez(:,1), 100, []);
    xs = reshape(Ez(:,2), 100, []);
    
    Ezs = reshape(Ez(:,3), 100, []);
    
    nxs = reshape(nx(:,3), 100, []);
    
    nys = reshape(ny(:,3), 100, []);
    
    nzs = reshape(nx(:,3), 100, []);
    
    uxs = reshape(ux(:,3), 100, []);
    
    uys = reshape(uy(:,3), 100, []);
    
    vols = reshape(vol_p(:,3), 100, []);
    
    
    figure(); surf(times, xs, Ezs);
    view(0, 90); shading interp, axis tight; xlabel('time (step)'); ylabel('location'); title('electric field');
    colorbar;
    
    figure(); surf(times, xs, nxs);
    view(0, 90); shading interp, axis tight; xlabel('time (step)'); ylabel('location'); title('orientation along x axis');
    colorbar;
    
    figure(); surf(times, xs, nys);
    view(0, 90); shading interp, axis tight; xlabel('time (step)'); ylabel('location'); title('orientation along y axis');
    colorbar;
    
    figure(); surf(times, xs, nzs);
    view(0, 90); shading interp, axis tight; xlabel('time (step)'); ylabel('location'); title('orientation along z axis');
    colorbar;
    
    figure(); surf(times, xs, uxs);
    view(0, 90); shading interp, axis tight; xlabel('time (step)'); ylabel('location'); title('ux');
    colorbar;
    
    figure(); surf(times, xs, uys);
    view(0, 90); shading interp, axis tight; xlabel('time (step)'); ylabel('location'); title('uy');
    colorbar;
    
    figure(); surf(times, xs, vols);
    view(0, 90); shading interp, axis tight; xlabel('time (step)'); ylabel('location'); title('voltage');
    colorbar;
    
    
    %% normalize
    Ezs = Ezs / max(max(abs(Ezs))) / 2;
    nxs = nxs / 2;
    nys = nys / 2;
    nzs = nzs / 2;
    uxs = uxs / max(max(abs(uxs))) / 2;
    uys = uys / max(max(abs(uys))) / 2;
    vols = vols / max(max(abs(vols))) / 2;
    
    sample_dT = 5e-4;
    L = max(max(xs));
    Adata_raw_train = [Ezs; nxs; nys; nzs; uxs; uys];
    n_da = size(Ezs, 1);
    n_kind = size(Adata_raw_train, 1) / n_da;
    
    train_input_sequence = zeros(size(Adata_raw_train));
    for k = 1:n_kind
        train_input_sequence(k:n_kind:end, :) = Adata_raw_train((k-1)*n_da+1:k*n_da, :);
    end
    train_input_sequence = train_input_sequence.';
    Input = vols.';
    x_ = xs(:, 1);
    N = size(x_, 1);
    
    % clearvars -except Adata Input sample_dT;
    % train_filename = strcat('LCD1d/LCD_data.mat');
    filename = [data_dir '/LCD_data.mat'];
    save(filename, 'train_input_sequence', 'Input', 'N', 'x_', 'sample_dT', 'L', '-v7.3');
    
    figure(); plot(times(1,:), nxs(10,:));




data_dir = test_dir;
    % Ex.dat: 電解のx成分
    % Ez.dat: 電解のz成分
    % nx.dat: 液晶の向きのｘ成分
    % ny.dat: 液晶の向きのｙ成分
    % nz.dat: 液晶の向きのｚ成分
    % pot.dat: 電位
    % vol_pix.dat: 画素電極の電位
    
    Ez = importdata(fullfile(data_dir, 'elec.dat')).data;
    % Ex = importdata(fullfile(test_dir, 'Ex.dat')).data;
    % Ez = importdata(fullfile(test_dir, 'Ez.dat')).data;
    nx = importdata(fullfile(data_dir, 'nx.dat')).data;
    ny = importdata(fullfile(data_dir, 'ny.dat')).data;
    nz = importdata(fullfile(data_dir, 'nz.dat')).data;
    ux = importdata(fullfile(data_dir, 'ux.dat')).data;
    uy = importdata(fullfile(data_dir, 'uy.dat')).data;
    % pot = importdata(fullfile(test_dir, 'pot.dat')).data;
    vol_p = importdata(fullfile(data_dir, 'volp.dat')).data;
    % elec = importdata(fullfile(test_dir, 'elec.dat')).data;
    
    
    times = reshape(Ez(:,1), 100, []);
    xs = reshape(Ez(:,2), 100, []);
    
    Ezs = reshape(Ez(:,3), 100, []);
    
    nxs = reshape(nx(:,3), 100, []);
    
    nys = reshape(ny(:,3), 100, []);
    
    nzs = reshape(nx(:,3), 100, []);
    
    uxs = reshape(ux(:,3), 100, []);
    
    uys = reshape(uy(:,3), 100, []);
    
    vols = reshape(vol_p(:,3), 100, []);
    
    
    figure(); surf(times, xs, Ezs);
    view(0, 90); shading interp, axis tight; xlabel('time (step)'); ylabel('location'); title('electric field');
    colorbar;
    
    figure(); surf(times, xs, nxs);
    view(0, 90); shading interp, axis tight; xlabel('time (step)'); ylabel('location'); title('orientation along x axis');
    colorbar;
    
    figure(); surf(times, xs, nys);
    view(0, 90); shading interp, axis tight; xlabel('time (step)'); ylabel('location'); title('orientation along y axis');
    colorbar;
    
    figure(); surf(times, xs, nzs);
    view(0, 90); shading interp, axis tight; xlabel('time (step)'); ylabel('location'); title('orientation along z axis');
    colorbar;
    
    figure(); surf(times, xs, uxs);
    view(0, 90); shading interp, axis tight; xlabel('time (step)'); ylabel('location'); title('ux');
    colorbar;
    
    figure(); surf(times, xs, uys);
    view(0, 90); shading interp, axis tight; xlabel('time (step)'); ylabel('location'); title('uy');
    colorbar;
    
    figure(); surf(times, xs, vols);
    view(0, 90); shading interp, axis tight; xlabel('time (step)'); ylabel('location'); title('voltage');
    colorbar;
    
    
    %% normalize
    Ezs = Ezs / max(max(abs(Ezs))) / 2;
    nxs = nxs / 2;
    nys = nys / 2;
    nzs = nzs / 2;
    uxs = uxs / max(max(abs(uxs))) / 2;
    uys = uys / max(max(abs(uys))) / 2;
    vols = vols / max(max(abs(vols))) / 2;
    
    sample_dT = 5e-4;
    L = max(max(xs));
    Adata_raw_test = [Ezs; nxs; nys; nzs; uxs; uys];
    n_da = size(Ezs, 1);
    n_kind = size(Adata_raw_test, 1) / n_da;
    
    test_input_sequence = zeros(size(Adata_raw_test));
    for k = 1:n_kind
        test_input_sequence(k:n_kind:end, :) = Adata_raw_test((k-1)*n_da+1:k*n_da, :);
    end
    
    test_input_sequence = test_input_sequence.';
    Input = vols.';
    x_ = xs(:, 1);
    N = size(x_, 1);
    
    % clearvars -except Adata Input sample_dT;
    % train_filename = strcat('LCD1d/LCD_data.mat');
    filename = [data_dir '/LCD_data.mat'];
    save(filename, 'test_input_sequence', 'Input', 'N', 'x_', 'sample_dT', 'L', '-v7.3');
    
    figure(); plot(times(1,:), nxs(10,:));


