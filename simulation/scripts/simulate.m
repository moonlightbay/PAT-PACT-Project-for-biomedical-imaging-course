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
% 设置项目根目录为当前文件夹
addpath('..\..\utils\k-wave-toolbox-version-1.4\k-Wave');  % 请将此路径替换为 k-Wave 工具箱的实际安装路径
% -------------------------------------------------------------------------
% 1. 仿真环境与网格设置
% -------------------------------------------------------------------------

% 数据保存路径
save_dir_data = 'pa_data';
if ~exist(save_dir_data, 'dir'), mkdir(save_dir_data); end


% 定义计算网格 (256 x 256)
Nx = 256;           % x方向网格数
Ny = 256;           % y方向网格数
dx = 0.1e-3;        % x方向网格间距 [m] (0.1 mm)
dy = 0.1e-3;        % y方向网格间距 [m] (0.1 mm)
kgrid = kWaveGrid(Nx, dx, Ny, dy);

% 定义介质属性
medium.sound_speed = 1540;  % 声速 [m/s] (人体软组织常用值)
medium.density = 1000;      % 密度 [kg/m^3]

% 设置时间数组
% 根据CFL条件自动计算时间步长，并设置总时长以覆盖整个区域
kgrid.makeTime(medium.sound_speed); 

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
% 3. 加载或生成初始声压分布 (Source)
% -------------------------------------------------------------------------

% 尝试加载图像，如果不存在则生成血管仿体
% 请将您的血管图像放置在 data/raw/ 目录下或修改此路径
% img_path = fullfile('..', '..', 'data', 'raw', 'vessel_mask.png'); 
img_path = '..\..\data\\raw\\output_MRA\\10001-TOF_ADAM_mip.png';
if exist(img_path, 'file')
    % 加载图像并调整大小为 Nx * Ny
    p0_image = loadImage(img_path); 
    if size(p0_image, 1) ~= Nx || size(p0_image, 2) ~= Ny
        p0_image = imresize(p0_image, [Nx, Ny]);
    end
    % loadImage 已归一化并反向映射，这里翻转回“血管亮、背景暗”
    p0 = 1 - p0_image;
    disp(['已加载图像: ', img_path]);
else
    % 生成血管仿体 (Vessel Phantom)
    disp('未找到指定图像，生成默认血管仿体...');
    % 使用 makeDisc 生成几个圆形代表血管截面
    p0 = zeros(Nx, Ny);
    p0 = p0 + makeDisc(Nx, Ny, Nx/2, Ny/2, 10);          % 中心大血管
    p0 = p0 + makeDisc(Nx, Ny, Nx/2+30, Ny/2-20, 6);     % 旁侧小血管
    p0 = p0 + makeDisc(Nx, Ny, Nx/2-20, Ny/2+40, 4);     % 旁侧微血管
    
    % 平滑处理以避免数值伪影 (参考示例使用 smooth)
    % 注意：如果遇到 MATLAB 自带 smooth 函数冲突，可改用 gaussianFilter
    if exist('smooth', 'file')
        p0 = smooth(p0, true); 
    else
        p0 = gaussianFilter(p0, [Nx, Ny], 1, true);
    end
end

source.p0 = p0;

% -------------------------------------------------------------------------
% 4. 执行 k-Wave 仿真
% -------------------------------------------------------------------------

disp('开始仿真...');

% 设置输入参数
% PlotSim: 是否实时显示波场
% RecordMovie: 是否录制波场传播视频
% PMLInside: false 表示 PML 层在网格外部，确保 256x256 区域完全有效 (参考示例)
input_args = {'PlotSim', true, 'RecordMovie', false, 'MovieName', 'pact_sim', 'PMLInside', false};

% 运行 2D 前向仿真
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});

disp('仿真完成。');

% -------------------------------------------------------------------------
% 5. 保存数据与结果展示
% -------------------------------------------------------------------------

% 保存仿真数据 (模拟 RF 信号)
% 文件名包含时间戳或索引，这里使用固定名称演示
save_filename_data = fullfile(save_dir_data, 'sim_data_circular.mat');
save(save_filename_data, 'sensor_data', 'kgrid', 'medium', 'sensor');
disp(['仿真数据已保存至: ', save_filename_data]);

% 可视化结果
figure('Name', 'PACT Simulation Results', 'Color', 'w', 'Position', [100, 100, 1000, 400]);

% 显示初始声压分布
subplot(1, 2, 1);
imagesc(kgrid.y_vec * 1e3, kgrid.x_vec * 1e3, source.p0);
axis image;
colormap(gray);
colorbar;
xlabel('y [mm]');
ylabel('x [mm]');
title('初始声压分布 (Source)');

% 显示传感器接收到的信号
subplot(1, 2, 2);
imagesc(1:num_sensor_points, kgrid.t_array * 1e6, sensor_data);
xlabel('传感器阵元索引');
ylabel('时间 [\mus]');
title('接收到的光声信号 (Sensor Data)');
colorbar;


