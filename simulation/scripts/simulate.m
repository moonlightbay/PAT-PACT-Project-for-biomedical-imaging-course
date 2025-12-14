% =========================================================================
% PACT 前向模拟仿真脚本
% 描述：
%   本脚本用于模拟光声计算机断层扫描 (PACT) 的前向过程。
%   输入：256x256 的血管图像掩膜（初始声压分布）。
%   设置：环形超声换能器阵列。
%   输出：换能器接收到的光声信号 (RF Data)。
%   参考：create_dataset(1).m 及 k-Wave 官方示例。
% =========================================================================

clear; clc; close all;

% =========================================================================
% 批量仿真参数设置区域
% =========================================================================

% 项目根目录(当前工作目录的上两级)
root_dir = fileparts(fileparts(pwd));

% 导入 k-Wave 工具箱路径
addpath(fullfile(root_dir, 'utils', 'k-wave-toolbox-version-1.4', 'k-Wave'));  % 请将此路径替换为 k-Wave 工具箱的实际安装路径

% 输入图像文件夹路径
input_image_dir = fullfile(root_dir, 'data',  'mask'); % 请根据实际情况修改
    
% 输入图像文件前缀
input_image_prefix = 'Mask'; % 请根据实际情况修改

% 索引范围 (包含起始和结束索引)
start_idx = 1517;
end_idx = 2520;

% 标签文件（初始光声压力分布）前缀
ground_truth_prefix = 'ground_truth';

% 仿真结果文件前缀
simulation_prefix = 'pa_data';

% PA数据输出文件夹路径
pa_data_output_dir = fullfile(root_dir, 'data', 'pa_data');

% 真实标签输出文件夹路径
ground_truth_output_dir = fullfile(root_dir, 'data', 'ground_truth');

% 创建输出文件夹
if ~exist(pa_data_output_dir, 'dir'), mkdir(pa_data_output_dir); end
if ~exist(ground_truth_output_dir, 'dir'), mkdir(ground_truth_output_dir); end

% -------------------------------------------------------------------------
% 1. 仿真环境与网格设置
% -------------------------------------------------------------------------


% 定义计算网格 (256 x 256)
Nx = 256;           % x方向网格数
Ny = 256;           % y方向网格数
dx = 0.1e-3;        % x方向网格间距 [m] (0.1 mm)
dy = 0.1e-3;        % y方向网格间距 [m] (0.1 mm)
kgrid = kWaveGrid(Nx, dx, Ny, dy);

% 定义介质属性
medium.sound_speed = 1540;  % 声速 [m/s] (人体软组织常用值)
medium.density = 1000;      % 密度 [kg/m^3]

% 手动设置时间数组，确保传感器数据尺寸为 128 x 2040
% 同时满足 CFL 稳定性条件 (cfl <= 0.3)
fs = 5.2e+07;       % 采样率 [Hz] (52 MHz - 满足 CFL 条件，≈ fs_max)
Nt = 2040;          % 时间步数
dt = 1 / fs;        % 时间步长 [s]
tarray = (0:1/fs:(Nt-1)/fs);  % 时间数组
kgrid.t_array = tarray;

% 验证 CFL 条件
cfl = medium.sound_speed * dt / min(dx, dy);
fprintf('CFL 数: %.4f\n', cfl);
if cfl <= 0.3
    fprintf('CFL 条件满足 (< 0.3)\n');
else
    fprintf('警告: CFL 条件不满足，建议降低采样率或调整网格\n');
end 

% -------------------------------------------------------------------------
% 2. 定义环形换能器阵列
% -------------------------------------------------------------------------
% 参考 k-Wave 官方示例: example_pr_2D_TR_circular_sensor.m

% 传感器几何参数
sensor_radius = (Nx/2 - 10) * dx;   % 环形阵列传感器半径 [m] (留出 10 个网格点的边缘余量)
num_sensor_points = 128;            % 传感器阵元数量
sensor_angle = 2*pi;                % 覆盖角度 (全环 2*pi)
sensor_pos = [0, 0];                % 环形阵列中心位置 [m] (对应网格中心)

% 创建笛卡尔坐标系下的环形传感器掩膜
% makeCartCircle 返回 2 x N 的矩阵，表示每个阵元的 (x, y) 坐标
sensor.mask = makeCartCircle(sensor_radius, num_sensor_points, sensor_pos, sensor_angle);

% 设置传感器的频率响应 (模拟真实探头特性，参考 create_dataset(1).m)
% 注意：官方示例 example_pr_2D_TR_circular_sensor.m 未包含此项，但为了更真实的 PACT 模拟保留
center_freq = 6.5e6;        % 中心频率 [Hz]
bandwidth = 70;             % 带宽 [%]
sensor.frequency_response = [center_freq, bandwidth];

% -------------------------------------------------------------------------
% 3. 批量处理循环
% -------------------------------------------------------------------------

fprintf('开始批量仿真，处理图像索引范围: %d 到 %d\n', start_idx, end_idx);

for idx = start_idx:end_idx
    % 构建当前图像文件名和路径
    % 假设文件名格式为: 前缀_0001.png
    image_filename = sprintf('%s_%04d.png', input_image_prefix, idx);
    img_path = fullfile(input_image_dir, image_filename);
    
    if ~exist(img_path, 'file')
        warning('图像文件不存在: %s', img_path);
        continue;
    end
    
    % 加载图像并调整大小为 Nx * Ny
    p0_image = loadImage(img_path); 
    if size(p0_image, 1) ~= Nx || size(p0_image, 2) ~= Ny
        p0_image = imresize(p0_image, [Nx, Ny]);
    end
    % loadImage 已归一化并反向映射，这里翻转回"血管亮、背景暗"
    p0 = 1 - p0_image;
    fprintf('已加载图像: %s\n', img_path);
    
    % 保存真实标签（初始光声压力分布）
    ground_truth_filename = sprintf('%s_%04d.mat', ground_truth_prefix, idx);
    ground_truth_path = fullfile(ground_truth_output_dir, ground_truth_filename);
    save(ground_truth_path, 'p0');
    
    % 设置源
    source.p0 = p0;

    % -----------------------------------------------------------------
    % 4. 执行 k-Wave 仿真
    % -----------------------------------------------------------------

    fprintf('开始仿真第 %d 个图像...\n', idx);

    % 设置输入参数
    % PlotSim: 是否实时显示波场 (批量处理时关闭)
    % RecordMovie: 是否录制波场传播视频 (批量处理时关闭)
    % PMLInside: false 表示 PML 层在网格外部，确保 256x256 区域完全有效 (参考示例)
    input_args = {'PlotSim', false, 'RecordMovie', false, 'PMLInside', false};

    % 运行 2D 前向仿真
    sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});

    fprintf('第 %d 个图像仿真完成。\n', idx);

    % -----------------------------------------------------------------
    % 5. 保存数据
    % -----------------------------------------------------------------

    % 保存仿真数据 (模拟 RF 信号)
    simulation_filename = sprintf('%s_%04d.mat', simulation_prefix, idx);
    simulation_path = fullfile(pa_data_output_dir, simulation_filename);
    save(simulation_path, 'sensor_data', 'kgrid', 'medium', 'sensor');
    fprintf('仿真数据已保存至: %s\n', simulation_path);

    % % 可选：显示当前处理结果 (每5个图像显示一次)
    % if mod(idx, 5) == 0
    %     figure('Name', sprintf('PACT Simulation Results - Image %d', idx), 'Color', 'w', 'Position', [100, 100, 1000, 400]);

    %     % 显示初始声压分布
    %     subplot(1, 2, 1);
    %     imagesc(kgrid.y_vec * 1e3, kgrid.x_vec * 1e3, source.p0);
    %     axis image;
    %     colormap(gray);
    %     colorbar;
    %     xlabel('y [mm]');
    %     ylabel('x [mm]');
    %     title('初始声压分布 (Source)');

    %     % 显示传感器接收到的信号
    %     subplot(1, 2, 2);
    %     imagesc(1:num_sensor_points, kgrid.t_array * 1e6, sensor_data);
    %     xlabel('传感器阵元索引');
    %     ylabel('时间 [\mus]');
    %     title('接收到的光声信号 (Sensor Data)');
    %     colorbar;
    % end

end  % 结束批量处理循环

fprintf('批量仿真完成！\n');


