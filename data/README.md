# 数据集说明文档 (Data Documentation)

本文档详细说明了本项目中生成和使用的数据集结构、文件格式以及生成流程。数据主要由 Python 脚本生成病灶模拟图像，再由 MATLAB (k-Wave) 脚本进行光声前向仿真生成。

## 1. 数据生成流程 (Pipeline)

整个数据生成流程如下：

1.  **血管掩膜预处理**：原始血管图像经过重命名和预处理放置在 `data/processed/renamed/` (作为输入源)。
2.  **病灶生成 (`pic_groundtruth.py`)**：
    *   读取血管掩膜。
    *   模拟脑卒中缺血（降低像素值）生成 SO2 分布图像 -> 存入 `data/mask/`。
    *   生成对应的病灶分割标签（正常/半暗带/核心） -> 存入 `data/label/`。
3.  **光声仿真 (`simulate.m`)**：
    *   读取 `data/mask/` 中的 SO2 图像作为初始声压分布 ($p_0$)。
    *   运行 k-Wave 2D 前向仿真。
    *   保存仿真得到的传感器数据 (RF Data) -> 存入 `data/pa_data/`。
    *   保存用于仿真的真实初始声压矩阵 -> 存入 `data/ground_truth/`。

---

## 2. 目录结构

```text
data/
├── mask/               # [输入] 模拟 SO2 分布的血管图像 (用于仿真输入)
├── label/              # [标签] 对应的病灶分割标签 (用于分割任务)
├── pa_data/            # [输出] k-Wave 仿真生成的传感器原始数据 (.mat)
└── ground_truth/       # [输出] 仿真实际使用的初始声压矩阵 (.mat)
```

---

## 3. 详细数据格式

### 3.1. 模拟 SO2 图像 (`data/mask/`)
由 `utils/pic_process/pic_groundtruth.py` 生成。

*   **文件格式**: `.png` (单通道灰度图)
*   **命名规范**: `Mask_XXXX.png` (例如 `Mask_1397.png`)
*   **内容描述**: 
    *   基于血管结构，模拟了脑卒中后的血氧饱和度 (SO2) 分布。
    *   **像素值含义**: 
        *   0: 背景
        *   高值 (~210-255): 正常血管区域
        *   中值 (~110-160): 半暗带 (Penumbra) 区域
        *   低值 (~30-60): 梗死核心 (Core) 区域
    *   **用途**: 作为 `simulate.m` 的输入，映射为初始声压分布 $p_0$。

### 3.2. 病灶分割标签 (`data/label/`)
由 `utils/pic_process/pic_groundtruth.py` 生成。

*   **文件格式**: `.png` (单通道灰度图，看起来可能是全黑的，需用代码读取数值)
*   **命名规范**: `Mask_XXXX_label.png`
*   **内容描述**: 
    *   像素值代表不同的组织类别。
    *   **类别定义**:
        *   `0`: 背景 (Background)
        *   `1`: 正常血管 (Normal Vessel)
        *   `2`: 半暗带 (Penumbra) - 缺血但可能挽救的区域
        *   `3`: 梗死核心 (Infarct Core) - 不可逆损伤区域
    *   **用途**: 用于训练语义分割模型，评估对病灶区域的识别能力。

### 3.3. 光声仿真数据 (`data/pa_data/`)
由 `simulation/scripts/simulate.m` 生成。

*   **文件格式**: `.mat` (MATLAB v7.3 或更高)
*   **命名规范**: `pa_data_XXXX.mat` (对应 `Mask_XXXX.png`)
*   **变量列表**:
    *   `sensor_data`: (矩阵) 传感器接收到的时域信号。尺寸通常为 `[num_sensors, num_time_steps]` (例如 128 x 2000+)。这是重建算法的输入。
    *   `kgrid`: (结构体) k-Wave 网格对象，包含 `Nx`, `Ny`, `dx`, `dy`, `t_array` 等时空参数。
    *   `medium`: (结构体) 介质属性，包含声速 `sound_speed` 和密度 `density`。
    *   `sensor`: (结构体) 传感器定义，包含 `mask` (位置坐标) 和频率响应参数。

### 3.4. 真实初始声压 (`data/ground_truth/`)
由 `simulation/scripts/simulate.m` 生成。

*   **文件格式**: `.mat`
*   **命名规范**: `ground_truth_XXXX.mat`
*   **变量列表**:
    *   `p0`: (256x256 double 矩阵) 仿真中实际使用的初始声压分布。
    *   **注意**: 该矩阵已经过预处理（调整大小、归一化、极性翻转等），代表了物理上的初始压力。
    *   **用途**: 作为图像重建任务的 "Ground Truth" (标签)，用于计算 PSNR/SSIM 或训练重建网络。

---

## 4. 注意事项

1.  **极性问题**: 
    *   `data/mask/` 中的 PNG 图片通常是黑色背景，白色(或灰色)血管。
    *   `simulate.m` 读取图片时，使用了 `loadImage` (k-Wave) 并进行了 `1 - p0` 操作，确保在仿真物理模型中，血管对应高声压 (1)，背景对应低声压 (0)。
    *   `data/ground_truth/` 中的 `p0` 矩阵是**血管为高值**的正确物理量。

2.  **坐标系**:
    *   仿真使用 k-Wave 的笛卡尔坐标系。
    *   传感器数据对应的是环形阵列。
