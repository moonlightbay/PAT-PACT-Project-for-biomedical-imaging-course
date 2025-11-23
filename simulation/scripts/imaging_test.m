% =========================================================================
% PACT 重建测试脚本
% 描述：
%   使用 k-Wave 的时间反演算法，将仿真得到的光声传感器数据重建为图像。
%   输入：simulate.m 保存的 sim_data_circular.mat。
%   输出：时间反演重建图像及原始信号的可视化。
% =========================================================================

clear; clc; close all;
addpath('..\..\utils\k-wave-toolbox-version-1.4\k-Wave');

% -------------------------------------------------------------------------
% 1. 载入仿真数据
% -------------------------------------------------------------------------

data_path = fullfile('pa_data', 'sim_data_circular.mat');
if ~isfile(data_path)
	error('未找到模拟数据文件：%s', data_path);
end

load(data_path, 'sensor_data', 'kgrid', 'medium', 'sensor');

% -------------------------------------------------------------------------
% 2. 构建与仿真一致的重建网格
% -------------------------------------------------------------------------

kgrid_recon = kWaveGrid(kgrid.Nx, kgrid.dx, kgrid.Ny, kgrid.dy);
kgrid_recon.setTime(kgrid.Nt, kgrid.dt);

% -------------------------------------------------------------------------
% 3. 将笛卡尔传感器阵列映射为连续环形掩膜
% -------------------------------------------------------------------------

cart_sensor_mask = sensor.mask;  % 2 x N，单位为米
sensor_radius = mean(sqrt(sum(cart_sensor_mask.^2, 1)));        % 平均半径 [m]
sensor_radius_grid = max(1, round(sensor_radius / kgrid_recon.dx));
binary_sensor_mask = makeCircle(kgrid_recon.Nx, kgrid_recon.Ny, ...
	kgrid_recon.Nx / 2 + 1, kgrid_recon.Ny / 2 + 1, sensor_radius_grid);

sensor_recon.mask = binary_sensor_mask;
sensor_recon.time_reversal_boundary_data = interpCartData(...
	kgrid_recon, sensor_data, cart_sensor_mask, binary_sensor_mask);

% -------------------------------------------------------------------------
% 4. 时间反演重建
% -------------------------------------------------------------------------

source_recon.p0 = 0;  % 时间反演需将初始声压清零
input_args = {'PMLInside', false, 'PlotSim', false};

disp('开始时间反演重建...');
p0_recon = kspaceFirstOrder2D(kgrid_recon, medium, source_recon, sensor_recon, input_args{:});
disp('重建完成。');

% -------------------------------------------------------------------------
% 5. 结果可视化
% -------------------------------------------------------------------------

figure('Name', 'PACT Reconstruction', 'Color', 'w', 'Position', [100, 100, 900, 400]);

subplot(1, 2, 1);
imagesc(1:size(sensor_data, 1), kgrid_recon.t_array * 1e6, sensor_data);
xlabel('传感器阵元索引');
ylabel('时间 [\mus]');
title('输入的光声信号');
colorbar;

subplot(1, 2, 2);
imagesc(kgrid_recon.y_vec * 1e3, kgrid_recon.x_vec * 1e3, p0_recon);
axis image;
colormap(gray);
colorbar;
xlabel('y [mm]');
ylabel('x [mm]');
title('时间反演重建结果');

