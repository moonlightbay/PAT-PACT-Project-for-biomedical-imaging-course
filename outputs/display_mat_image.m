% display_mat_image.m
% 读取并显示指定的.mat文件和.png文件为图像
% 使用左右方向键可以切换图像索引 (0001 到 2520)

function display_mat_image
    % 全局变量存储当前索引
    global current_index;
    
    % 初始化索引
    if isempty(current_index)
        current_index = 1;
    end
    
    % 设置文件路径（使用 fullfile 函数自动处理路径分隔符）
    % 1. 旧的重建结果
    mat_file = fullfile('..', 'data', 'recon', 'pa_data_%04d.mat');
    % 2. 新的TR重建结果 (imaging_test生成)
    tr_file = fullfile('..', 'data', 'tr', 'tr_result_%04d.mat');
    % 3. 真实标签
    truth_file = fullfile('..', 'data', 'ground_truth', 'ground_truth_%04d.mat');
    
    % 创建figure（调整为2x3布局，变宽以容纳3列）
    h_fig = figure('Name', 'PACT重建结果对比', 'Position', [50, 50, 1600, 800]);
    
    % 设置键盘回调
    set(h_fig, 'KeyPressFcn', @(src, event) keyPressCallback(src, event, mat_file, tr_file, truth_file));
    
    % 初始显示
    updateDisplay(mat_file, tr_file, truth_file);
    
    % 显示使用说明
    fprintf('=== 使用说明 ===\n');
    fprintf('按 ← 键显示上一张图像\n');
    fprintf('按 → 键显示下一张图像\n');
    fprintf('当前图像索引: %04d\n', current_index);
end

function keyPressCallback(src, event, mat_file, tr_file, truth_file)
    global current_index;
    
    % 检查按键
    if strcmp(event.Key, 'leftarrow')
        % 左箭头键：上一张
        current_index = current_index - 1;
        if current_index < 1
            current_index = 2520;  % 循环到最后一个
        end
        updateDisplay(mat_file, tr_file, truth_file);
    elseif strcmp(event.Key, 'rightarrow')
        % 右箭头键：下一张
        current_index = current_index + 1;
        if current_index > 2520
            current_index = 1;  % 循环到第一个
        end
        updateDisplay(mat_file, tr_file, truth_file);
    end
end

function updateDisplay(mat_file, tr_file, truth_file)
    global current_index;
    
    % 生成当前文件路径
    [path, name, ext] = fileparts(mat_file);
    mat_path = fullfile(path, sprintf([name ext], current_index));
    
    [path, name, ext] = fileparts(tr_file);
    tr_path = fullfile(path, sprintf([name ext], current_index));
    
    [path, name, ext] = fileparts(truth_file);
    truth_path = fullfile(path, sprintf([name ext], current_index));
    
    fprintf('------------------------------------------------\n');
    fprintf('索引: %04d\n', current_index);
    
    % 1. 读取旧重建数据 (Recon)
    img_data = zeros(256, 256);
    if exist(mat_path, 'file')
        try
            data = load(mat_path);
            field_names = fieldnames(data);
            % 假设第一个变量是图像数据
            temp_img = data.(field_names{1});
            temp_img = double(abs(temp_img));
            % 归一化
            temp_img(temp_img < 0) = 0;
            if max(temp_img(:)) > 0
                temp_img = temp_img / max(temp_img(:));
            end
            img_data = temp_img;
        catch ME
            warning('无法读取旧重建文件 %s: %s', mat_path, ME.message);
        end
    else
        % warning('旧重建文件不存在: %s', mat_path);
    end
    
    % 2. 读取新TR重建数据 (Imaging Test)
    tr_img = zeros(256, 256);
    if exist(tr_path, 'file')
        try
            data = load(tr_path);
            % 优先查找 p0_recon 变量
            if isfield(data, 'p0_recon')
                temp_img = data.p0_recon;
            else
                % 否则查找第一个数值型矩阵
                vars = fieldnames(data);
                found = false;
                for i = 1:length(vars)
                    if isnumeric(data.(vars{i})) && ndims(data.(vars{i})) == 2
                        temp_img = data.(vars{i});
                        found = true;
                        break;
                    end
                end
                if ~found
                    temp_img = zeros(256, 256);
                end
            end
            
            temp_img = double(abs(temp_img));
            % 归一化 (虽然脚本里已经做了，这里再做一次保险)
            temp_img(temp_img < 0) = 0;
            if max(temp_img(:)) > 0
                temp_img = temp_img / max(temp_img(:));
            end
            tr_img = temp_img;
        catch ME
            warning('无法读取TR重建文件 %s: %s', tr_path, ME.message);
        end
    else
        % warning('TR重建文件不存在: %s', tr_path);
    end
    
    % 3. 读取真实标签 (Ground Truth)
    truth_img = zeros(256, 256);
    if exist(truth_path, 'file')
        try
            truth_data = load(truth_path);
            truth_names = fieldnames(truth_data);
            temp_img = truth_data.(truth_names{1});
            truth_img = double(abs(temp_img));
            % 归一化
            if max(truth_img(:)) > 0
                truth_img = truth_img / max(truth_img(:));
            end
        catch ME
            warning('无法读取真实标签文件 %s: %s', truth_path, ME.message);
        end
    else
        warning('真实标签文件不存在: %s', truth_path);
    end
    
    % 应用0.45-0.75范围处理（缺血半暗带）
    lower_threshold = 0.45;
    upper_threshold = 0.75;
    
    img_data_thresh = img_data;
    img_data_thresh(img_data_thresh < lower_threshold) = 0;
    img_data_thresh(img_data_thresh > upper_threshold) = 0;
    
    tr_img_thresh = tr_img;
    tr_img_thresh(tr_img_thresh < lower_threshold) = 0;
    tr_img_thresh(tr_img_thresh > upper_threshold) = 0;
    
    truth_img_thresh = truth_img;
    truth_img_thresh(truth_img_thresh < lower_threshold) = 0;
    truth_img_thresh(truth_img_thresh > upper_threshold) = 0;
    
    % 清除figure
    clf;
    
    % 显示6个图像（2x3布局）
    
    % --- 第一行：完整图像 ---
    
    % 1. 真实标签
    subplot(2, 3, 1);
    imagesc(truth_img);
    colormap(gca, hot);
    colorbar;
    axis image;
    title(sprintf('真实标签 (Ground Truth)\n[%04d]', current_index), 'FontSize', 11, 'FontWeight', 'bold');
    xlabel('X'); ylabel('Y');
    
    % 2. TR重建
    subplot(2, 3, 2);
    imagesc(tr_img);
    colormap(gca, hot);
    colorbar;
    axis image;
    title(sprintf('TR重建 (Imaging Test)\n[%04d]', current_index), 'FontSize', 11, 'FontWeight', 'bold');
    xlabel('X'); ylabel('Y');
    
    % 3. ATT-DUNET重建
    subplot(2, 3, 3);
    imagesc(img_data);
    colormap(gca, hot);
    colorbar;
    axis image;
    title(sprintf('ATT-DUNET重建\n[%04d]', current_index), 'FontSize', 11, 'FontWeight', 'bold');
    xlabel('X'); ylabel('Y');
    
    % --- 第二行：阈值处理后的图像 ---
    
    % 4. 真实标签 (阈值)
    subplot(2, 3, 4);
    imagesc(truth_img_thresh);
    colormap(gca, hot);
    colorbar;
    axis image;
    title(sprintf('真实标签 (0.45-0.75)'), 'FontSize', 10);
    xlabel('X'); ylabel('Y');
    
    % 5. TR重建 (阈值)
    subplot(2, 3, 5);
    imagesc(tr_img_thresh);
    colormap(gca, hot);
    colorbar;
    axis image;
    title(sprintf('TR重建 (0.45-0.75)'), 'FontSize', 10);
    xlabel('X'); ylabel('Y');
    
    % 6. ATT-DUNET重建 (阈值)
    subplot(2, 3, 6);
    imagesc(img_data_thresh);
    colormap(gca, hot);
    colorbar;
    axis image;
    title(sprintf('ATT-DUNET重建 (0.45-0.75)'), 'FontSize', 10);
    xlabel('X'); ylabel('Y');
    
    % 添加总标题
    sgtitle(sprintf('PACT 重建结果对比 - 索引 %04d (← → 切换)', current_index), ...
        'FontSize', 14, 'FontWeight', 'bold');
    
    % 刷新显示
    drawnow;
end
