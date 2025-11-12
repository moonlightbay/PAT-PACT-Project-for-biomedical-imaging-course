import torch
import os
import scipy.io as sio
from dunet import DUNet
from att_dunet import Att_DUNet
import numpy as np

# 设置CUDA设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载训练好的模型
model = Att_DUNet()
model_path = "best_attdunet_model.pth"
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()  # 设置模型为评估模式

# 定义测试数据集路径
test_pa_data_dir = "test_pa_data"
test_recon_dir = "test_recon"

# 创建输出目录
if not os.path.exists(test_recon_dir):
    os.makedirs(test_recon_dir)

# 获取所有测试文件
test_files = [f for f in os.listdir(test_pa_data_dir) if f.endswith('.mat')]

# 逐个文件进行预测
with torch.no_grad():
    for i, test_file in enumerate(test_files):
        # 读取.mat文件
        mat_data = sio.loadmat(os.path.join(test_pa_data_dir, test_file))
        mat_input = mat_data['sensor_data_2D']  # 假设文件中的数据键是'input'

        # 转换为Tensor，并将其转移到GPU
        mat_input_tensor = torch.tensor(mat_input, dtype=torch.float32).unsqueeze(0) # 加一维batch
        mat_input_tensor = mat_input_tensor.unsqueeze(0).to(device)

        # 模型推理
        output = model(mat_input_tensor)

        # 将预测结果转为NumPy数组
        output_np = output.cpu().numpy().squeeze()  # 去掉batch维度

        # 保存结果为.mat文件
        output_filename = os.path.join(test_recon_dir, test_file)
        sio.savemat(output_filename, {"p0": output_np})

        print(f"file {test_file} saved")

print("all saved to test_recon folder")
