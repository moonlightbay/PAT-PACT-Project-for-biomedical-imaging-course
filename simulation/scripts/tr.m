% =========================================================================
% PACT 批量重建脚本
% 描述：
%   使用 k-Wave 的时间反演算法，将仿真得到的光声传感器数据批量重建为图像。
%   输入：data/pa_data/pa_data_xxxx.mat
%   输出：data/tr/tr_result_xxxx.mat
% =========================================================================

clear; clc; close all;

% -------------------------------------------------------------------------
% 0. 路径与参数设置
% -------------------------------------------------------------------------

% 获取当前脚本所在路径，并推导项目根目录
script_dir = fileparts(mfilename('fullpath'));
root_dir = fileparts(fileparts(script_dir)); % 回退两级到项目根目录

% 添加 k-Wave 工具箱路径
addpath(fullfile(root_dir, 'utils', 'k-wave-toolbox-version-1.4', 'k-Wave'));

% 输入数据目录
pa_data_dir = fullfile(root_dir, 'data', 'pa_data');

% 输出数据目录
tr_output_dir = fullfile(root_dir, 'data', 'tr');
if ~exist(tr_output_dir, 'dir')
    mkdir(tr_output_dir);
end

% 设置处理的索引范围
start_idx = 1;    % 请根据实际情况修改起始索引
end_idx = 1;      % 请根据实际情况修改结束索引

fprintf('项目根目录: %s\n', root_dir);
fprintf('输入目录: %s\n', pa_data_dir);
fprintf('输出目录: %s\n', tr_output_dir);
fprintf('开始批量重建，索引范围: %d - %d\n', start_idx, end_idx);

% -------------------------------------------------------------------------
% 批量处理循环
% -------------------------------------------------------------------------

for idx = start_idx:end_idx
    
    % 1. 构建文件路径
    data_filename = sprintf('pa_data_%04d.mat', idx);
    data_path = fullfile(pa_data_dir, data_filename);
    
    output_filename = sprintf('tr_result_%04d.mat', idx);
    output_path = fullfile(tr_output_dir, output_filename);
    
    % 检查输入文件是否存在
    if ~isfile(data_path)
        warning('未找到数据文件：%s，跳过。', data_filename);
        continue;
    end
    
    fprintf('正在处理 [%d/%d]: %s ... \n', idx, end_idx, data_filename);
    
    % 2. 载入仿真数据
    % 加载 sensor_data, kgrid, medium, sensor
    load(data_path, 'sensor_data', 'kgrid', 'medium', 'sensor');
    
    % 3. 构建与仿真一致的重建网格
    kgrid_recon = kWaveGrid(kgrid.Nx, kgrid.dx, kgrid.Ny, kgrid.dy);
    kgrid_recon.setTime(kgrid.Nt, kgrid.dt);
    
    % 4. 将笛卡尔传感器阵列映射为连续环形掩膜
    cart_sensor_mask = sensor.mask;  % 2 x N，单位为米
    sensor_radius = mean(sqrt(sum(cart_sensor_mask.^2, 1)));        % 平均半径 [m]
    sensor_radius_grid = max(1, round(sensor_radius / kgrid_recon.dx));
    binary_sensor_mask = makeCircle(kgrid_recon.Nx, kgrid_recon.Ny, ...
        kgrid_recon.Nx / 2 + 1, kgrid_recon.Ny / 2 + 1, sensor_radius_grid);
    
    sensor_recon.mask = binary_sensor_mask;
    sensor_recon.time_reversal_boundary_data = interpCartData(...
        kgrid_recon, sensor_data, cart_sensor_mask, binary_sensor_mask);
    
    % 5. 时间反演重建
    source_recon.p0 = 0;  % 时间反演需将初始声压清零
    input_args = {'PMLInside', false, 'PlotSim', false};
    
    % 执行重建
    p0_recon = kspaceFirstOrder2D(kgrid_recon, medium, source_recon, sensor_recon, input_args{:});
    
    % 6. 后处理：确保结果为0-1的灰度图
    % 将负值设为0（光声压力不能为负）
    p0_recon = max(p0_recon,0);
    % 归一化到 [0, 1] 范围
    if max(p0_recon(:)) > 0
        p0_recon = p0_recon / max(p0_recon(:));
    end
    
    % 7. 保存结果
    save(output_path, 'p0_recon', 'kgrid_recon');
    fprintf('  -> 重建完成，已保存至: %s\n', output_filename);

end

fprintf('批量重建全部完成。\n');

% -------------------------------------------------------------------------
% 结果可视化 (仅显示最后一次处理的结果，用于检查)
% -------------------------------------------------------------------------
if exist('p0_recon', 'var')
    figure('Name', 'Last Reconstruction Result', 'Color', 'w', 'Position', [100, 100, 900, 400]);

    subplot(1, 2, 1);
    imagesc(1:size(sensor_data, 1), kgrid_recon.t_array * 1e6, sensor_data);
    xlabel('传感器阵元索引');
    ylabel('时间 [\mus]');
    title(['输入信号: ', data_filename]);
    colorbar;

    subplot(1, 2, 2);
    imagesc(kgrid_recon.y_vec * 1e3, kgrid_recon.x_vec * 1e3, p0_recon);
    axis image;
    colormap("hot");
    colorbar;
    xlabel('y [mm]');
    ylabel('x [mm]');
    title('时间反演重建结果');
end
