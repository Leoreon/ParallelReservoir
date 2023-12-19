
clear
% L = 22; N = 64;
L = 22; N = 840;
% load('KS_L50_N_128_dps20000.mat');
% load('KS_L22_N_64_dps20000.mat');
load(['KS_L' num2str(L) '_N_' num2str(N) '_dps20000.mat']);

% test_input_sequence = trajectories_true.';
[n_steps, n_data] = size(test_input_sequence);
exporttype = 'mp4';
save_steps = 300;
% save_steps = n_steps;
    
%% step 1
% グラフデータを生成するコードをここに書く
%% step 2
fig = figure; % Figure オブジェクトの生成
% 図を描画するコード(軸ラベルや範囲など)をここに書く
frames(save_steps) = struct('cdata', [], 'colormap', []); % 各フレームの画像データを格納する配列
for i = 1:save_steps % 動画の長さは100フレームとする
    % 図を更新するコードをここに書く
    space = L*[0:1/n_data:1-1/n_data];
    plot(space, test_input_sequence(i, :).');

    % xticks(0:0.5:1); yticks(-3:3:3); 
    xticks(L*[0:0.5:1]); yticks(-3:3:3); 
    % set(gcf, 'color', 'w');
    xlim(L*[0 1]); ylim([-3 3]); xlabel('x'); ylabel('y');
    fontsize(16, 'points');
    drawnow; % 描画を確実に実行させる
    frames(i) = getframe(fig); % 図を画像データとして得る
end
%% step 3
switch exporttype % 今回は出力方法を exporttype 変数で指定することにする
    case 'mp4' % 普通の動画の場合
        video = VideoWriter(['KS_L' num2str(L) '_N' num2str(N) '_' num2str(save_steps) 'steps_' 'snapshot.mp4'], 'MPEG-4'); % ファイル名や出力形式などを設定
        open(video); % 書き込むファイルを開く
        writeVideo(video, frames); % ファイルに書き込む
        close(video); % 書き込むファイルを閉じる
    case 'gif'
        filename = 'filename.gif'; % ファイル名
        for i = 1:100
            [A, map] = rgb2ind(frame2im(frames(i)), 256); % 画像形式変換
            if i == 1
                imwrite(A, map, filename, 'gif', 'DelayTime', 1/30); % 出力形式(30FPS)を設定
            else
                imwrite(A, map, filename, 'gif', 'DelayTime', 1/30, 'WriteMode', 'append'); % 2フレーム目以降は“追記“の設定も必要
            end
        end
end
